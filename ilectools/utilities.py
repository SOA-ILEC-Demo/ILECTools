'''
Constants defined here:
    VAL_COLS: the main value columns: amount and policy actuals, 2015vbt, exposure
    TAB_KEYS: names of key values for mortality rates. qx should only vary by these
    NEW_NAMES: dictionary from original name in lowercawse to name in form  [amount|policy]_[item]
        where item is one of actual, exposure, 7580e, 2001vbt, 2008vbt, 2008vbtlu, 2015vbt,
        or one of the central moment fields
        e.g. amount_2015vbt for the 2015vbt tabular figures by face amount.
        Those values are deliberately not called expecteds here because they are not a mean
        and nobody expects them.

'''

import io
import matplotlib.pyplot as plt
import base64

VAL_COLS = [f'{m}_{i}'
            for m in ['amount', 'policy']
            for i in ['actual','exposure','2015vbt']
            ]

TAB_KEYS = 'age_basis,smoker_status,gender,issue_age,duration'.split(',')
NEW_NAMES = {
    'number_of_deaths': 'policy_actual',
    'death_claim_amount': 'amount_actual',
    'policies_exposed': 'policy_exposure',
    'amount_exposed': 'amount_exposure',
    'expected_death_qx7580e_by_amount': 'amount_7580e',
    'expected_death_qx2001vbt_by_amount': 'amount_2001vbt',
    'expected_death_qx2008vbt_by_amount': 'amount_2008vbt',
    'expected_death_qx2008vbtlu_by_amount': 'amount_2008vbtlu',
    'expected_death_qx2015vbt_by_amount': 'amount_2015vbt',
    'expected_death_qx7580e_by_policy': 'policy_7580e',
    'expected_death_qx2001vbt_by_policy': 'policy_2001vbt',
    'expected_death_qx2008vbt_by_policy': 'policy_2008vbt',
    'expected_death_qx2008vbtlu_by_policy': 'policy_2008vbtlu',
    'expected_death_qx2015vbt_by_policy': 'policy_2015vbt',
    'cen2momp1wmi_byamt': 'amount_cen2momp1wmi',
    'cen2momp2wmi_byamt': 'amount_cen2momp2wmi',
    'cen3momp1wmi_byamt': 'amount_cen3momp1wmi',
    'cen3momp2wmi_byamt': 'amount_cen3momp2wmi',
    'cen3momp3wmi_byamt': 'amount_cen3momp3wmi',
    'cen2momp2wmi_bypol': 'policy_cen2momp2wmi',
    'cen3momp3wmi_bypol': 'policy_cen3momp3wmi',
    'cen2momp1_byamt': 'amount_cen2momp1',
    'cen2momp2_byamt': 'amount_cen2momp2',
    'cen3momp1_byamt': 'amount_cen3momp1',
    'cen3momp2_byamt': 'amount_cen3momp2',
    'cen3momp3_byamt': 'amount_cen3momp3',
    'cen2momp2_bypol': 'policy_cen2momp2',
    'cen3momp3_bypol': 'policy_cen3momp3'}


def figure_base64():
    """Get base64 encoding from current figure plt.gcf() for display in html"""
    # https://stackoverflow.com/questions/56361175/why-is-picture-annotation-cropped-when-displaying-or-saving-figure/56370990#56370990
    figfile = io.BytesIO()
    plt.savefig(figfile, format='png', bbox_inches='tight')
    figfile.seek(0)  # rewind to beginning of file
    # must change from byte string to normal string
    return str(base64.b64encode(figfile.getvalue()), 'utf-8')
