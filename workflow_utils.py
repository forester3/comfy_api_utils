def find_node_ids_from_connections(workflow_nodes_dict):
    found = {}
    node_by_id = {str(nid): node for nid, node in workflow_nodes_dict.items()}

    for nid, node in workflow_nodes_dict.items():
        ntype = node.get("class_type") or node.get("type")
        nid_str = str(nid)

        if ntype == "KSampler":
            inputs = node.get("inputs", {})
            for key, label in [("positive", "PositivePrompt_TextEncode"),
                               ("negative", "NegativePrompt_TextEncode")]:
                ref = inputs.get(key)
                if isinstance(ref, list) and len(ref) == 2:
                    ref_id = str(ref[0])
                    ref_node = node_by_id.get(ref_id)
                    if ref_node and (ref_node.get("class_type") in ["CLIPTextEncode", "CLIPTextEncodeSDXL"] or
                                     ref_node.get("type") in ["CLIPTextEncode", "CLIPTextEncodeSDXL"]):
                        found[label] = ref_id
            found["KSampler"] = nid_str

        elif ntype in ["EmptyLatentImage", "SaveImage", "CheckpointLoaderSimple", "CLIPSetLastLayer"]:
            found[ntype] = nid_str

    return found
