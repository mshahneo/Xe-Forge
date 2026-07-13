module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %dpas = transform.structured.match ops{["xegpu.dpas"]} in %root : (!transform.any_op) -> !transform.any_op
    %aval = transform.get_operand %dpas[0] : (!transform.any_op) -> !transform.any_value
    %aload = transform.xegpu.get_load_op %aval : (!transform.any_value) -> !transform.any_op
    %bval = transform.get_operand %dpas[1] : (!transform.any_op) -> !transform.any_value
    %bload = transform.xegpu.get_load_op %bval : (!transform.any_value) -> !transform.any_op
    transform.xegpu.set_anchor_layout %aload sg_layout = [8, 1] sg_data = [32, 32] inst_data = [8, 16] : !transform.any_op
    transform.xegpu.set_anchor_layout %bload sg_layout = [1, 8] sg_data = [32, 32] inst_data = [16, 16] : !transform.any_op
    transform.xegpu.set_anchor_layout %dpas index = 0 sg_layout = [8, 1] sg_data = [32, 32] inst_data = [8, 16] : !transform.any_op
    transform.xegpu.set_anchor_layout %dpas index = 1 sg_layout = [1, 8] sg_data = [32, 32] inst_data = [16, 16] : !transform.any_op
    transform.xegpu.set_anchor_layout %dpas index = 2 sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] : !transform.any_op
    %store = transform.structured.match ops{["xegpu.store_nd"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.xegpu.set_anchor_layout %store sg_layout = [8, 8] sg_data = [32, 32] inst_data = [8, 16] : !transform.any_op
    transform.yield
  }
}
