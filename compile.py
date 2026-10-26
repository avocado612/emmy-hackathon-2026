#!/usr/bin/env python3
"""Kernel IR -> PTX compiler.

    python compile.py path/to/<kernel>.ir.json > out.ptx

stdout is reserved for PTX only, so anything for humans goes to stderr.
"""

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from emmy.compiler.graph import Graph

@dataclass
class KernelSpec:
    # launch info + GPU kernel IR
    name: str          # kernel name
    body: tuple        # given emmy IR
    params: list       # arg_order, one .u64 param each
    grid: tuple
    block: tuple
    smem_bytes: int

def load_kernel(path):
    with open(path) as f:
        graph = Graph.from_dict(json.load(f))
    # we compile KernelOp only because it is what compiles to PTX
    for node in graph.nodes.values():
        if type(node.op).__name__ == "KernelOp":
            return node.op
    sys.exit(f"{path}: no KernelOp found")

def load_spec(ir_path):
    #Load the KernelOp (GPU kernel IR)
    gpu_kernel_ir = load_kernel(ir_path)
    # load the Meta (Kernel Launch Code)
    meta = json.loads((Path(ir_path).parent / "meta.json").read_text())

    for meta_kernel_spec in meta["kernels"]:
        if meta_kernel_spec["entry"] == gpu_kernel_ir.name:
            return KernelSpec(gpu_kernel_ir.name,
                              gpu_kernel_ir.body,
                              meta_kernel_spec["arg_order"],
                              tuple(meta_kernel_spec["grid"]),
                              tuple(meta_kernel_spec["block"]),
                              meta_kernel_spec["smem_bytes"]
                             )
    sys.exit(f"{ir_path}: no kernel '{gpu_kernel_ir.name}' in meta.json")

class Emitter:
    """Collects PTX lines and hands out fresh virtual registers.

    Registers are never reused: ptxas does the real register allocation, so we
    only count how many of each kind we asked for. The .reg declarations need
    those final counts, so they are written last (see render).
    """

    # PTX register names emmy uses : PTX reg types
    REG_TYPES = {"r":".b32", 
                 "p":".pred",
                 "f":".f32",
                 "rd":".b64"
                }

    def __init__(self, spec):
        self.spec = spec
        self.ptx_code = []                                 
        self.reg_count = {k: 0 for k in self.REG_TYPES}     # create [r:0, p:0, f:0, rd:0]
        self.buffers = {}                                # buffer name -> pointer register
        self.vars = {}                                   # IR variable name -> register

    # _________________generic helpers_____________________

    def add_line(self, line):
        self.ptx_code.append("        " + line)

    def label(self, name):
        self.ptx_code.append(name + ":")

    def new_reg(self, reg_name):
        n = self.reg_count[reg_name]
        self.reg_count[reg_name] += 1
        return f"%{reg_name}{n}"

    # _________________Prologue___________________________

    def header(self):
        return [".version 8.0", ".target sm_89", ".address_size 64", ""]

    def signature(self):
        signature_ptx = [f".visible .entry {self.spec.name}("]
        for i, name in enumerate(self.spec.params):
            if i < (len(self.spec.params) - 1):
                signature_ptx.append(f"        .param .u64 param_{name},")
            else:
                signature_ptx.append(f"        .param .u64 param_{name}")
        signature_ptx.append(")")
        return signature_ptx

    def init_gpu_regs(self):
        lines = []
        for reg_name, ptx_type in self.REG_TYPES.items():
            num_reg = max(1, self.reg_count[reg_name])
            lines.append(f"        .reg {ptx_type}       %{reg_name}<{num_reg}>;")
        return lines

    def load_input_output_pointers(self):
        for name in self.spec.params:
            rd_reg_ptr = self.new_reg("rd")
            global_reg_ptr = self.new_reg("rd")
            self.add_line(f"ld.param.u64 {rd_reg_ptr}, [param_{name}];")
            self.add_line(f"cvta.to.global.u64 {global_reg_ptr}, {rd_reg_ptr}; \n")
            self.buffers[name] = global_reg_ptr

    def init_thread_ids(self):
        self.tid = self.new_reg("r")
        blockIdx_reg = self.new_reg("r")
        blockDimx_reg = self.new_reg("r")
        self.gid = self.new_reg("r")

        self.add_line(f"mov.u32 {self.tid}, %tid.x;")
        self.add_line(f"mov.u32 {blockIdx_reg}, %ctaid.x;")
        self.add_line(f"mov.u32 {blockDimx_reg}, %ntid.x;")
        #multiply-add: d = a * b + c.
        self.add_line(f"mad.lo.u32 {self.gid}, {blockDimx_reg}, {blockIdx_reg}, {self.tid}; \n")


    #Threads with gid >= N (N = product of the Tile extents) skip to DONE
    def bounds_guard(self, tile):
        n_cells = 1
        for axis in tile.axes:
            n_cells *= axis.extent.value
        skip_thread = self.new_reg("p")
        self.add_line(f"setp.ge.u32 {skip_thread}, {self.gid}, {n_cells};")
        self.add_line(f"@{skip_thread} bra DONE; \n")

    # def decode_axes(self, tile):
    #     """Split gid into the Tile's axis variables, innermost axis first.

    #     Row-major: for axes (a0, a1) with a1 extent E, a1 = gid % E and a0 = gid / E.
    #     The outermost axis needs no modulo; the innermost needs no division
    #     before its own modulo. Results go into self.vars for index expressions.
    #     """
    #     rest = self.gid
    #     for i, axis in reversed(list(enumerate(tile.axes))):
    #         if i == 0:
    #             self.vars[axis.name] = rest        # outermost: what is left is the index
    #             break
    #         extent = axis.extent.value
    #         axis_reg = self.new_reg("r")
    #         self.add_line(f"rem.u32 {axis_reg}, {rest}, {extent};")
    #         self.vars[axis.name] = axis_reg
    #         quotient = self.new_reg("r")
    #         self.add_line(f"div.u32 {quotient}, {rest}, {extent};")
    #         rest = quotient
    #     self.add_line("")

    # _________________WALK THE EMMY IR_________________


    def emit_expr(self, expr):
        # 1, Compile the ast expression tree
        # 2. add the ptx code to ptx_code
        # 3. return the newly assigned register to hold the expression value.

        #Every node gets a fresh register; no constant folding (ptxas cleans up).
        kind = type(expr).__name__

        # children node
        if kind == "Var":
            return self.vars[expr.name]

        # 1. mvoing values into registers because there is 
        # no limit for how many ptx registers we can use
        # 2. we also want to use check diffreent d-types as well

        # children node
        elif kind == "Literal":
            reg = self.new_reg("r")
            self.add_line(f"mov.u32 {reg}, {expr.value};")
            return reg

        # parent node - can't be a child node
        elif kind == "BinaryExpr":
            left = self.emit_expr(expr.left)
            right = self.emit_expr(expr.right)
            reg = self.new_reg("r")
            if expr.op == "+":
                self.add_line(f"add.u32 {reg}, {left}, {right};")
            elif expr.op == "*":
                self.add_line(f"mul.lo.u32 {reg}, {left}, {right};")
            else:
                sys.exit(f"unsupported BinaryExpr op '{expr.op}'")
            return reg

        else:
            sys.exit(f"emit_expr: unsupported expression {kind}")

    def get_address(self, buffer, index_reg, nbytes=4):
        #1. calculate the address of uint8_t buffer[index]
        #2. return the reg that holds the address
        offset_reg = self.new_reg("rd")
        self.add_line(f"mul.wide.u32 {offset_reg}, {index_reg}, {nbytes};") # cell_size * index

        final_byte_addr_reg = self.new_reg("rd")
        self.add_line(f"add.s64 {final_byte_addr_reg}, {self.buffers[buffer]}, {offset_reg};")

        return final_byte_addr_reg

    # _________________Statements_________________________
    def emit_stmt(self, stmt):
        # type(stmt) returns it's class, and __name__ makes it a string
        kind = type(stmt).__name__ 
        # matches an attirubte in this Emitter Class named "emit_{kind}"
        handler = getattr(self, f"emit_{kind}", None)
        if handler is None:
            sys.exit(f"handler for this stmt doesn't exit: '{kind}'")

        # emit_Load
        # emit_Assign
        # emit_Write
        handler(stmt)

    def emit_Load(self, stmt):
        # one scalar load per name; vector (v2/v4) loads come later
        if len(stmt.names) != 1:
            sys.exit(f"emit_Load: vector load of {len(stmt.names)} not supported yet")

        idx = self.emit_expr(stmt.index[0])
        addr = self.get_address(stmt.input, idx, stmt.dtype.nbytes)
        new_reg = self.new_reg("f")
        self.add_line(f"ld.global.f32 {new_reg}, [{addr}];")
        self.vars[stmt.names[0]] = new_reg

    # elementwise op name -> PTX template ({d} = result, {a}/{b} = arguments)
    ELEMENTWISE = {
        "add"       : "add.f32 {d}, {a}, {b};",
        "subtract"  : "sub.f32 {d}, {a}, {b};",
        "multiply"  : "mul.f32 {d}, {a}, {b};",
        "divide"    : "div.rn.f32 {d}, {a}, {b};",
        "maximum"   : "max.f32 {d}, {a}, {b};",
        "relu"      : "max.f32 {d}, {a}, 0f00000000;",   # max(x, 0.0)
        "copy"      : "mov.f32 {d}, {a};",
        "sqrt"      : "sqrt.rn.f32 {d}, {a};",
    }

    def emit_Assign(self, stmt):
        op_name = stmt.op.name
        ptx_template = self.ELEMENTWISE.get(op_name)
        if ptx_template is None:
            sys.exit(f"elementwise op '{op_name}' not supported")
        args = [self.vars[a] for a in stmt.args]
        dst_reg = self.new_reg("f")
        # unary ops have no {b} in their template, so the extra kwarg is ignored
        if len(args) > 1:
            b = args[1] 
        else:
            b = None
        self.add_line(ptx_template.format(d=dst_reg, a=args[0], b=b))
        self.vars[stmt.name] = dst_reg

    def emit_Write(self, stmt):

        reg_list_to_write = [self.vars[v] for v in stmt.values]
        reg_idx = self.emit_expr(stmt.index[0])
        base_buffer = stmt.output
        addr_reg = self.get_address(base_buffer, reg_idx, stmt.value_dtype.nbytes)
        if stmt.atomic == True:
            # atomic adds go one value at a time: red.global.add has no vector form here
            for i, val in enumerate(reg_list_to_write):
                self.add_line(f"red.global.add.f32 [{addr_reg}+{i * stmt.value_dtype.nbytes}], {val};")
        elif len(reg_list_to_write) == 1:
            self.add_line(f"st.global.f32 [{addr_reg}], {reg_list_to_write[0]};")
        elif len(reg_list_to_write) in (2, 4):
            # EX: st.global.v2.f32 [%rd_addr], {%f_v0__u0, %f_v0__u1};
            self.add_line(f"st.global.v{len(reg_list_to_write)}.f32 [{addr_reg}], {{{', '.join(reg_list_to_write)}}};")
        else:
            sys.exit(f"emit_Write: group of {len(reg_list_to_write)} values not supported")

    def render(self):
        # Build the body first: the .reg counts are only known once every
        # instruction has asked for its registers.

        # ___Prologue___

        self.load_input_output_pointers()
        self.init_thread_ids()
        tile = self.spec.body[0]
        # print(tile, file=sys.stderr)
        self.bounds_guard(tile)
        # One axis only: the axis index is the global thread id.
        self.vars[tile.axes[0].name] = self.gid
        # self.decode_axes(tile)

        # ___Walk____
        print("\n\n", file=sys.stderr)
        for stmt in tile.body:
            print(stmt, file=sys.stderr)
            self.ptx_code.append("\n")
            self.emit_stmt(stmt)

        # ___Emit___


        out = self.header()
        out += self.signature() + ["{"]
        out += self.init_gpu_regs() + [""]
        out += self.ptx_code + ["DONE:", "        ret;", "}"]

        return "\n".join(out) + "\n" # REMEBER TO keep it a string!

def main():
    if len(sys.argv) != 2:
        sys.exit("should only have 2 inputs: argv[0]=compile.py, argv[1]=IR")
    spec = load_spec(sys.argv[1])

    # ____debug graph IR on the terminal______
    print(f"---- IR body of {spec.name} ----", file=sys.stderr)
    pprint(spec.body, stream=sys.stderr, width=100)  # pprint breaks lines and indents nested statements

    em = Emitter(spec)
    sys.stdout.write(em.render()) # writes to -> out.ptx


if __name__ == "__main__":
    main()
