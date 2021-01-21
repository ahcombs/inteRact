class eAxes:
    xAxis, yAxis, zAxis = range(3)

class eTurn:
    learner, simulator = range(2)

class eEPA:
    evaluation, potency, activity = range(3)

    evaluationSelf, potencySelf, activitySelf,\
    evaluationAction, potencyAction, activityAction,\
    evaluationOther, potencyOther, activityOther = range(9)

    fundamental, tau = range(2)

class eIdentityParse:
    identity, maleEvaluation, malePotency, maleActivity,\
    femaleEvaluation, femalePotency, femaleActivity, institution = range(8)

class eAgentListBoxParam:
    identity, maleSentiment, femaleSentiment, institution = range(4)

class eInstitutions:
    gender, institution, undefined = range(3)

class eInteractants:
    agent, client = range(2)

class eGender:
    male, female = range(2)

class eGenderKey:
    anyGender, male, female = range(3)

class eGui:
    simulator, interactive = range(2)

class eRect:
    fromLeft, fromBottom, fractionOfX, fractionOfY = range(4)