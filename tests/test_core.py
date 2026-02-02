# clue-codebert
from CLUE.core import CLUE
from CLUE.utils import EvaluationUtil
import json
best_actual_weights = {'we': 0.809949398089146, 'wr': 0.190050601910854, 'wn': 0.7868036728153308, 'wa': 0.10425542769201852, 'wm': 0.10894089949265066, 'wat': 0.5942232195411594, 'wan': 0.4057767804588406, 'wmp': 0.11660127933232939, 'wmn': 0.7306275085450178, 'wmt': 0.15277121212265277, 'wpt': 0.050258460422135706, 'wpn': 0.9497415395778643, 'wrt': 0.1556958305042492, 'wrq': 0.22047580654418472, 'wrn': 0.6238283629515661}
with open('./test_files/reference.json','r',encoding='utf-8') as f:
    reference = json.load(f)
with open('./test_files/candidate_human.json','r',encoding='utf-8') as f:
    candidate = json.load(f)
print("calculate clue-codebert ...")
clue_sim, clue_simce, clue_simca, clue_simcm, clue_simcr = 0, 0, 0, 0, 0
EvaluationUtil.set_base_model("codebert-base")
clue = CLUE(reference,candidate,EvaluationUtil.calc_similarity_batch("codebert-base"))
clue.setCLUEParams(
    we = best_actual_weights['we'], wr = best_actual_weights['wr'], 
    wn = best_actual_weights['wn'], wa = best_actual_weights['wa'], wm = best_actual_weights['wm'],
    wat = best_actual_weights['wat'], wan = best_actual_weights['wan'],
    wmn = best_actual_weights['wmn'], wmt = best_actual_weights['wmt'], wmp = best_actual_weights['wmp'],
    wpt = best_actual_weights['wpt'], wpn = best_actual_weights['wpn'],
    wrt = best_actual_weights['wrt'], wrq = best_actual_weights['wrq'], wrn = best_actual_weights['wrn']
)
clue.RunCLUE()
clue_sim = clue.getCLUE()
clue_simce = clue.getSimCE()
clue_simca = clue.getSimCA()
clue_simcm = clue.getSimCM()
clue_simcr = clue.getSimCR()

print(f"clue : {clue_sim}")
print(f"clue-class : {clue_simce}")
print(f"clue-attribute: {clue_simca}")
print(f"clue-method: {clue_simcm}")
print(f"clue-relation: {clue_simcr}")