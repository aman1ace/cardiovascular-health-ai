import numpy as np


def predict_health(age, hr_before, hr_after, max_hr, sleep, risk_model, score_model):

    recovery = hr_before - hr_after
    stress = max_hr / (220 - age)

    X_input = np.array([[age, sleep, recovery, stress]])

    score = score_model.predict(X_input)[0]
    risk_prob = risk_model.predict_proba(X_input)[0][1]
    risk = risk_model.predict(X_input)[0]

    # -----------------------------
    # SAFETY OVERRIDE
    # -----------------------------
    # If heart rate after exercise is equal or higher than before,
    # it means poor recovery / abnormal recovery response.
    if hr_after >= hr_before:
        risk_prob = max(risk_prob, 0.75)
        risk = 1
        final_status = "At Risk"

    else:
        # -----------------------------
        # FINAL DECISION LOGIC
        # -----------------------------
        if score < 40 or risk_prob > 0.6:
            final_status = "At Risk"
        elif score < 65:
            final_status = "Moderate"
        else:
            final_status = "Healthy"

    return {
        "Health Score": round(score, 2),
        "Risk (Model)": "At Risk" if risk == 1 else "Fit",
        "Risk Probability": round(risk_prob * 100, 2),
        "Final Decision": final_status,
        "recovery": round(recovery, 2),
        "stress": round(stress, 2)
    }
