"""Run the supplied D:/CTLE/Model.py with outputs contained in this repo.

Use CTLE_REFERENCE_EQUALIZER=perfect to change only the equalizer class.
Only offset 1 is trained so the comparison finishes quickly.
"""

import os
from pathlib import Path


source = Path(r"D:\CTLE\Model.py").read_text(encoding="utf-8")
output_root = Path(__file__).resolve().parent / "result" / "reference_f_comparison"
equalizer = os.environ.get("CTLE_REFERENCE_EQUALIZER", "f_layer")
if equalizer not in ("f_layer", "perfect"):
    raise ValueError("CTLE_REFERENCE_EQUALIZER must be f_layer or perfect")

output_dir = (str(output_root / equalizer).replace("\\", "/") + "/")
source = source.replace('path = "data_model/"', f'path = {output_dir!r}')
source = source.replace('name_Tx = "./data/Txdata.txt"',
                        'name_Tx = "D:/CTLE/data/Txdata.txt"')
source = source.replace('channel_path = "channel/"',
                        'channel_path = "D:/CTLE/channel/"')
source = source.replace('offset_candidates = list(range(sps))',
                        'offset_candidates = [1]')
source = source.replace('    return data.reshape(-1, 5000, 1).astype(np.float32)',
                        '    return data[:len(data) // 5000 * 5000].reshape(-1, 5000, 1).astype(np.float32)')
source = source.replace('    x, ref = inputs\n',
                        '    x = inputs[0]\n    ref = inputs[1]\n')
if equalizer == "perfect":
    source = source.replace(
        'from Layer import CTLELayer, FixedFIRChannelLayer, NormalizeRMSLayer, F_Layer',
        'from Layer import CTLELayer, FixedFIRChannelLayer, NormalizeRMSLayer, F_Layer, Layer_perfect',
    )
    source = source.replace('data_equalized = F_Layer(',
                            'data_equalized = Layer_perfect(')
    source = source.replace('        fft_length=f_layer_fft_length,\n', '')
    source = source.replace('"F_Layer": F_Layer,',
                            '"F_Layer": F_Layer, "Layer_perfect": Layer_perfect,')
    source = source.replace('\nprint_f_layer_weights(model)\n',
                            '\nprint("equalizer =", model.get_layer("rx_f_layer").__class__.__name__)\n')

exec(compile(source, r"D:\CTLE\Model.py", "exec"), {"__name__": "__main__"})
