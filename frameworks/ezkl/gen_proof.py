import os
import ezkl
import json
import numpy as np
import argparse

def gen_proof(output_folder, data_path , model_path):
    compiled_model_path = os.path.join(output_folder, 'network.compiled')
    settings_path = os.path.join(output_folder, 'settings.json') 
    witness_path = os.path.join(output_folder, 'witness.json')

    proof_path = os.path.join(output_folder, 'proof.json')

    pk_path = os.path.join(output_folder, 'test.pk')
    vk_path = os.path.join(output_folder, 'test.vk')


    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "public"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    assert res == True
    res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources", scales=[2,7])
    assert res == True

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    res = ezkl.get_srs(settings_path)

    # now generate the witness file
    res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    with open(witness_path, "r") as f:
        wit = json.load(f)

    with open(settings_path, "r") as f:
        setting = json.load(f)


    prediction_array = []
    for value in wit["outputs"]:
        for field_element in value:
            prediction_array.append(ezkl.vecu64_to_float(field_element, setting['model_output_scales'][0]))
    pred = np.argmax([prediction_array])

    res = ezkl.mock(witness_path, compiled_model_path)
    assert res == True

    res = ezkl.setup(
            compiled_model_path,
            vk_path,
            pk_path,
        )


    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # Generate the proof
    proof = ezkl.prove(
            witness_path,
            compiled_model_path,
            pk_path,
            proof_path,
            "single",
        )
    #print(proof)
    assert os.path.isfile(proof_path)
    print ('Proof Gen')
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate proof for a given model and data.")
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    parser.add_argument('--data', type=str, required=True, help='Data file path')
    parser.add_argument('--model', type=str, required=True, help='Model file path')

    args = parser.parse_args()

    pred = gen_proof(args.output, args.data, args.model)
    
    print(f"Prediction: {pred}")