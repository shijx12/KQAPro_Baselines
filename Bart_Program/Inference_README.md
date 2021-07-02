# Inference BART Program

    from Bart_Program.inference import Inference
    model_name_or_path = "../KQAPro_ckpt/program_ckpt"
    kb_json_file = "../KQA-Pro-v1.0/kb.json"
    inferencer = Inference(model_name_or_path, kb_json_file)
    inferencer.run("who is the prime minister of india?")

    # result are: ['Find(India)<b>Relate(country<c>backward)<b>FilterConcept(human)<b>What()']