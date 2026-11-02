import numpy as np


def coagulation_subscore(platelets):
    if (platelets > 150) or np.isnan(platelets):
        return 0
    elif (platelets > 100):
        return 1
    elif (platelets > 50):
        return 2
    elif (platelets > 20):
        return 3
    else:
        return 4


def liver_subscore(bilirubin):
    if (bilirubin < 1.2) or np.isnan(bilirubin):
        return 0
    elif (bilirubin <= 1.9):
        return 1
    elif (bilirubin < 6):
        return 2
    elif (bilirubin < 12):
        return 3
    else:
        return 4


def gcs_subscore(gcs):
    if (gcs > 14) or np.isnan(gcs):
        return 0
    elif (gcs > 12):
        return 1
    elif (gcs > 9):
        return 2
    elif (gcs > 5):
        return 3
    else:
        return 4


def renal_subscore(creatinine, urine_output):
    if (creatinine < 1.2):
        return 0
    elif (creatinine < 2):
        return 1
    elif (creatinine < 3.5):
        return 2
    elif (creatinine < 5) or (urine_output < 500):
        return 3
    elif (creatinine >= 5) or (urine_output < 200):
        return 4
    else:
        return 0


def cardiovascular_subscore(MAP, norepinephrine, epinephrine, dobutamine, dopamine):
    if (np.isnan(MAP) and np.isnan(norepinephrine) and np.isnan(epinephrine) and np.isnan(dobutamine) and
            np.isnan(dopamine)):
        return 0
    elif (dopamine > 15 or epinephrine > 0.1 or norepinephrine > 0.1):
        return 4
    elif (dopamine > 5 or (epinephrine < 0.1 and epinephrine > 0) or (norepinephrine < 0.1 and norepinephrine > 0)):
        return 3
    elif ((dopamine <= 5 and dopamine > 0) or dobutamine > 0):
        return 2
    elif (MAP < 70):
        return 1
    else:
        return 0


def respiratory_subscore(PaO2_FiO2, mech_vent_cpap):
    if (PaO2_FiO2 < 100) and (mech_vent_cpap == 1):
        return 4
    elif (PaO2_FiO2 < 200) and (mech_vent_cpap == 1):
        return 3
    elif (PaO2_FiO2 < 300):
        return 2
    elif (PaO2_FiO2 < 400):
        return 1
    else:
        return 0


def calc_sofa(components):
    platelets = components["Platelets_10_3_per_ml"]
    bilirubin = components["Total_Bilirubin_mg_per_dL"]
    gcs = components["GCS"]
    creatinine = components["Creatinine_mg_per_dL"]
    urine_output = components["Urine_output_mL"]
    MAP = components["MAP"]
    norepinephrine = components["Norepinephrine_μg_kg_min"]
    epinephrine = components["Epinephrine_μg_kg_min"]
    dobutamine = components["Dobutamine_any_dose"]
    dopamine = components["Dopamine_μg_kg_min"]
    pao2_fio2 = components["PaO2_FiO2"]
    mech_vent_cpap = components["mech_vent_cpap"]

    subscores = dict()
    subscores["coagulation"] = coagulation_subscore(platelets)
    subscores["liver"] = liver_subscore(bilirubin)
    subscores["GCS"] = gcs_subscore(gcs)
    subscores["renal"] = renal_subscore(creatinine, urine_output)
    subscores["cardiovascular"] = cardiovascular_subscore(MAP, norepinephrine, epinephrine, dobutamine, dopamine)
    subscores["respiratory"] = respiratory_subscore(pao2_fio2, mech_vent_cpap)

    sofa_score = float(sum(subscores.values()))

    return sofa_score
