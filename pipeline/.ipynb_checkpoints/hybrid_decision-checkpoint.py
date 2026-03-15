import pandas as pd

def apply_hybrid(df_classifier, df_reg, pRef):

    df_hybrid = pd.concat([df_classifier, df_reg], axis=1)

    df_hybrid["final_pred"] = df_hybrid["pred_classifier"]

    mask = df_hybrid["prob_classifier"] < pRef

    df_hybrid.loc[mask, "final_pred"] = df_hybrid.loc[mask, "reg_class"]

    return df_hybrid