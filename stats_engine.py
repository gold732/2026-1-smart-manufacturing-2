import numpy as np
import pandas as pd
from scipy.stats import shapiro, boxcox
from scipy.special import gamma

# ==========================================
# 1. 품질 통계 불편화 계수 및 상수 보정 모듈
# ==========================================

def calc_unbiased_const(const_name, n):
    """
    표본 크기에 따른 품질관리용 불편화 상수를 계산하거나 테이블에서 참조합니다.
    """
    try:
        d_table = pd.read_csv('unbiased_control_chart.csv')
    except Exception:
        if const_name == 'c4':
            if n <= 1: return 1.0
            return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
        elif const_name == 'd2':
            return 3.4873 + 0.0250141 * n - 0.00009823 * (n ** 2)
        return 1.0

    n = int(n)
    if const_name == 'd2':
        if n < 51:
            return float(d_table['d2'].iloc[n - 2])
        return 3.4873 + 0.0250141 * n - 0.00009823 * (n ** 2)
    elif const_name == 'c4':
        return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
    return 1.0


def unbiased_coefficient_fallback(coef_name, m):
    """
    관리도 한계선 산출을 위한 내장 상수를 매핑합니다.
    """
    table = {
        'A2': {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308},
        'A3': {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975},
        'B3': {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284},
        'B4': {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716},
        'D3': {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223},
        'D4': {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777},
        'd2': {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
    }
    if coef_name in table and m in table[coef_name]:
        return table[coef_name][m]
    return 1.0


# ==========================================
# 2. 정규성 검정 및 변환 모듈
# ==========================================

def run_normality_test(data_array):
    """
    Shapiro-Wilk 정규성 검정을 수행하여 p-value와 정규성 만족 여부를 반환합니다.
    """
    if len(data_array) < 3:
        return 1.0, True
    stat, p = shapiro(data_array)
    return float(p), bool(p >= 0.05)


def apply_box_cox(data_array, lsl_orig=None, usl_orig=None):
    """
    Box-Cox 변환을 수행하고, 입력된 규격 한계선(LSL/USL)을 동일 척도로 동시 변환합니다.
    """
    data_series = pd.Series(data_array)
    if (data_series <= 0).any():
        shift_val = abs(data_series.min()) + 1.0
        data_to_transform = data_series + shift_val
        if lsl_orig is not None: lsl_orig += shift_val
        if usl_orig is not None: usl_orig += shift_val
    else:
        data_to_transform = data_series

    transformed_data, lambda_val = boxcox(data_to_transform)
    
    lsl_trans = lsl_orig
    usl_trans = usl_orig
    
    if lambda_val == 0:
        if lsl_orig is not None and lsl_orig > 0: lsl_trans = np.log(lsl_orig)
        if usl_orig is not None and usl_orig > 0: usl_trans = np.log(usl_orig)
    else:
        if lsl_orig is not None and lsl_orig > 0: lsl_trans = (lsl_orig**lambda_val - 1) / lambda_val
        if usl_orig is not None and usl_orig > 0: usl_trans = (usl_orig**lambda_val - 1) / lambda_val
        
    return transformed_data, lsl_trans, usl_trans, float(lambda_val)


# ==========================================
# 3. 공정능력분석 연산 모듈 (tab3 7개 변수 반환 언패킹 구조 교정)
# ==========================================

def calculate_capability_metrics(df, sg_col, val_col, lsl, usl, method="Pooled Standard Deviation"):
    """
    군내 및 전체 표준편차를 기반으로 단기(Cp, Cpk) 및 장기(Pp, Ppk) 공정능력지수를 계산하여 7개의 변수로 반환합니다.
    """
    df_clean = df.dropna(subset=[sg_col, val_col])
    x_bar = df_clean[val_col].mean()
    
    sg_groups = df_clean.groupby(sg_col)[val_col]
    sg_stds = sg_groups.std(ddof=1).fillna(0)
    sg_ranges = sg_groups.apply(lambda x: x.max() - x.min())
    sg_sizes = sg_groups.size()
    
    m_size = int(sg_sizes.mode().iloc[0]) if not sg_sizes.empty else 5
    
    # 1. 전체 변동 산출
    sigma_hat = df_clean[val_col].std(ddof=1)
    c4_overall = calc_unbiased_const('c4', len(df_clean))
    sigma_overall = sigma_hat / c4_overall if c4_overall else sigma_hat
    
    # 2. 군내 변동 산출 방식 분기
    if method == "Pooled Standard Deviation":
        sigma_p = np.sqrt(np.sum(sg_stds**2) / len(sg_stds))
        subgroup_delta_d = len(df_clean) - len(sg_stds) + 1
        c4_within = calc_unbiased_const('c4', subgroup_delta_d)
        sigma_within = sigma_p / c4_within if c4_within else sigma_p
    else:
        r_bar = sg_ranges.mean()
        d2 = calc_unbiased_const('d2', m_size)
        sigma_within = r_bar / d2 if d2 else r_bar

    # 3. 공정능력지수 연산
    if lsl is not None and usl is not None:
        Cp = (usl - lsl) / (6 * sigma_within)
        Cpk = min((usl - x_bar) / (3 * sigma_within), (x_bar - lsl) / (3 * sigma_within))
        Pp = (usl - lsl) / (6 * sigma_overall)
        Ppk = min((usl - x_bar) / (3 * sigma_overall), (x_bar - lsl) / (3 * sigma_overall))
    else:
        Cp, Cpk, Pp, Ppk = np.nan, np.nan, np.nan, np.nan
        
    return Cp, Cpk, Pp, Ppk, x_bar, sigma_within, sigma_overall


# ==========================================
# 4. 계량형 관리도 연산 모듈
# ==========================================

def generate_value_chart_data(df, sg_col, val_col, chart_type='Xbar-R', window=3):
    """
    Xbar-R, Xbar-s, I-MR 관리도용 관리한계선 데이터프레임을 생성합니다.
    """
    df_clean = df.dropna(subset=[sg_col, val_col])
    sg = pd.DataFrame()
    sg['Xbar'] = df_clean.groupby(sg_col)[val_col].mean()
    sg['n_i'] = df_clean.groupby(sg_col)[val_col].size()
    subgroup_indices = sg.index
    m_size = int(sg['n_i'].mode().iloc[0]) if not sg['n_i'].empty else 5

    if chart_type == 'Xbar-R':
        sg['R'] = df_clean.groupby(sg_col)[val_col].apply(lambda x: x.max() - x.min())
        Xbar_bar = sg['Xbar'].mean()
        R_bar = sg['R'].mean()
        
        A2 = unbiased_coefficient_fallback('A2', m_size)
        D3 = unbiased_coefficient_fallback('D3', m_size)
        D4 = unbiased_coefficient_fallback('D4', m_size)
        
        chart1 = pd.DataFrame({'point': sg['Xbar'], 'CL': Xbar_bar, 'LCL': Xbar_bar - A2 * R_bar, 'UCL': Xbar_bar + A2 * R_bar}, index=subgroup_indices)
        chart2 = pd.DataFrame({'point': sg['R'], 'CL': R_bar, 'LCL': D3 * R_bar, 'UCL': D4 * R_bar}, index=subgroup_indices)
        return chart1, chart2

    elif chart_type == 'Xbar-s':
        sg['s'] = df_clean.groupby(sg_col)[val_col].std(ddof=1).fillna(0)
        Xbar_bar = sg['Xbar'].mean()
        s_bar = sg['s'].mean()
        
        A3 = unbiased_coefficient_fallback('A3', m_size)
        B3 = unbiased_coefficient_fallback('B3', m_size)
        B4 = unbiased_coefficient_fallback('B4', m_size)
        
        chart1 = pd.DataFrame({'point': sg['Xbar'], 'CL': Xbar_bar, 'LCL': Xbar_bar - A3 * s_bar, 'UCL': Xbar_bar + A3 * s_bar}, index=subgroup_indices)
        chart2 = pd.DataFrame({'point': sg['s'], 'CL': s_bar, 'LCL': B3 * s_bar, 'UCL': B4 * s_bar}, index=subgroup_indices)
        return chart1, chart2

    elif chart_type == 'I-MR':
        df_sort = df_clean.set_index(sg_col).sort_index()
        Xbar = df_sort[val_col].mean()
        MR_i = df_sort[val_col].rolling(window=window).apply(lambda x: x.max() - x.min(), raw=True)
        MR_bar = MR_i.dropna().mean()
        
        D3 = unbiased_coefficient_fallback('D3', window)
        D4 = unbiased_coefficient_fallback('D4', window)
        d2 = unbiased_coefficient_fallback('d2', window)
        if d2 == 0 or d2 is None: d2 = 1.128
        
        chart1 = pd.DataFrame({'point': df_sort[val_col], 'CL': Xbar, 'LCL': Xbar - 3 * MR_bar / d2, 'UCL': Xbar + 3 * MR_bar / d2}, index=df_sort.index)
        chart2 = pd.DataFrame({'point': MR_i, 'CL': MR_bar, 'LCL': D3 * MR_bar, 'UCL': D4 * MR_bar}, index=df_sort.index)
        return chart1, chart2
        
    return None, None


# ==========================================
# 5. 계수형 관리도 연산 모듈 (tab1 및 tab5 2개 객체 언패킹 튜플 구조 완벽 대응)
# ==========================================

def generate_count_chart(df_raw, sg_name, n_i, var_name, chart_type='NP'):
    """
    NP, P, C, U 관리도의 동적 관리한계선을 계산하며 UI 언패킹 호환을 위해 2개의 차트 객체 구조로 반환합니다.
    """
    data = df_raw.set_index(sg_name)
    count_chart = pd.DataFrame(index=data.index)
    
    if chart_type == 'NP':
        np_bar = data[var_name].sum() / len(data)
        p_bar = data[var_name].sum() / data[n_i].sum()
        
        count_chart['point'] = data[var_name]
        count_chart['CL'] = np_bar
        count_chart['LCL'] = np_bar - 3 * np.sqrt(np_bar * (1 - p_bar))
        count_chart['UCL'] = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))

    elif chart_type == 'P':
        p_bar = data[var_name].sum() / data[n_i].sum()
        
        count_chart['point'] = data[var_name] / data[n_i]
        count_chart['CL'] = p_bar
        count_chart['LCL'] = p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / data[n_i])
        count_chart['UCL'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / data[n_i])

    elif chart_type == 'C':
        c_bar = data[var_name].mean()
        
        count_chart['point'] = data[var_name]
        count_chart['CL'] = c_bar
        count_chart['LCL'] = c_bar - 3 * np.sqrt(c_bar)
        count_chart['UCL'] = c_bar + 3 * np.sqrt(c_bar)

    elif chart_type == 'U':
        u_bar = data[var_name].sum() / data[n_i].sum()
        
        count_chart['point'] = data[var_name] / data[n_i]
        count_chart['CL'] = u_bar
        count_chart['LCL'] = u_bar - 3 * np.sqrt(u_bar / data[n_i])
        count_chart['UCL'] = u_bar + 3 * np.sqrt(u_bar / data[n_i])
        
    count_chart['LCL'] = count_chart['LCL'].clip(lower=0)
    
    # tab1 및 tab5의 chart1, chart2 = engine.generate_count_chart(...) 튜플 분할 언패킹 구조 완벽 방어 보정
    return count_chart, None
