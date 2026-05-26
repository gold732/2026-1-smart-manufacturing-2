import numpy as np
import pandas as pd
from scipy.stats import shapiro, boxcox
from scipy.special import gamma

def calc_unbiased_const(const_name, n):
    """강의록 08_공정능력분석 15~16페이지의 수학적 근사식 및 불편화 상수 테이블 계산"""
    if n <= 1:
        return 1.0
    if const_name == 'd2':
        if n < 51:
            try:
                d_table = pd.read_csv('unbiased_control_chart.csv')
                return float(d_table['d2'].iloc[int(n)-2])
            except Exception:
                return 3.4873 + 0.0250141 * n - 0.00009823 * (n ** 2)
        return 3.4873 + 0.0250141 * n - 0.00009823 * (n ** 2)
    elif const_name == 'c4':
        return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
    return 1.0

def unbiased_coefficient_fallback(coef_name, m):
    """강의록 09_통계적공정관리 4~5페이지 제어 차트 상수 테이블 복구 조치"""
    table = {
        'A2': {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373},
        'A3': {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099},
        'D3': {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136},
        'D4': {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864},
        'B3': {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.030, 7: 0.118, 8: 0.185},
        'B4': {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815},
        'd2': {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847}
    }
    try:
        d_table = pd.read_csv('unbiased_control_chart.csv')
        if coef_name in d_table.columns and 2 <= m <= 25:
            return d_table[coef_name].iloc[m - 2]
    except Exception:
        pass
    if coef_name in table and m in table[coef_name]:
        return table[coef_name][m]
    return 1.0 if coef_name in ['d2', 'B4', 'D4', 'A2', 'A3'] else 0.0

def run_normality_test(values):
    p = shapiro(values)[1]
    return p, p >= 0.05

def analyze_process_capability(df, sg_col, val_col, lsl, usl, method="Pooled Standard Deviation"):
    """강의록 16~17페이지 장/단기 변동 분석 모델 수식 구현 (옵션 변환 반영)"""
    mean_sg = df.groupby(sg_col)[val_col].mean()
    sigma_sg = df.groupby(sg_col)[val_col].std().fillna(0)
    
    x_bar = df[val_col].mean()
    sigma_hat = df[val_col].std(ddof=1)
    
    # 장기 변동 산출 (Overall)
    c4_overall = calc_unbiased_const('c4', len(df))
    sigma_overall = sigma_hat / c4_overall if c4_overall > 0 else sigma_hat
    
    # 단기 변동 산출 (Within - 사용자 지정 선택 파라미터 매핑)
    subgroup_sizes = df[sg_col].value_counts()
    m_size = int(subgroup_sizes.mode().iloc[0]) if not subgroup_sizes.mode().empty else 2
    
    if method == "Pooled Standard Deviation":
        sigma_p = np.sqrt(np.sum(sigma_sg**2) / len(sigma_sg))
        c4_within = calc_unbiased_const('c4', len(df) - len(sigma_sg) + 1)
        sigma_within = sigma_p / c4_within if c4_within > 0 else sigma_p
    else:
        # 부분군 범위 평균(R-bar) 방식 분기 처리 구현
        r_sg = df.groupby(sg_col)[val_col].max() - df.groupby(sg_col)[val_col].min()
        r_bar = r_sg.mean()
        d2_const = unbiased_coefficient_fallback('d2', m_size)
        sigma_within = r_bar / d2_const if d2_const > 0 else sigma_hat

    Cp = (usl - lsl) / (6 * sigma_within) if sigma_within > 0 else 0
    Cpk = min((usl - x_bar) / (3 * sigma_within), (x_bar - lsl) / (3 * sigma_within)) if sigma_within > 0 else 0
    
    Pp = (usl - lsl) / (6 * sigma_overall) if sigma_overall > 0 else 0
    Ppk = min((usl - x_bar) / (3 * sigma_overall), (x_bar - lsl) / (3 * sigma_overall)) if sigma_overall > 0 else 0
    
    return {
        "x_bar": x_bar, "sigma_overall": sigma_overall, "sigma_within": sigma_within,
        "Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk
    }

def apply_box_cox(values, lsl, usl):
    transformed_vals, lambda_val = boxcox(values)
    def transform_value(x, lmbda):
        if lmbda == 0: return np.log(x)
        return (x**lmbda - 1) / lmbda
    return transformed_vals, transform_value(lsl, lambda_val), transform_value(usl, lambda_val), lambda_val

def generate_value_chart_data(data, sg_col, val_col, chart_type='Xbar-R', window=3):
    sg = pd.DataFrame()
    sg['Xbar'] = data.groupby(sg_col)[val_col].mean()
    sg['n_i'] = data[sg_col].value_counts()
    subgroup_indices = sg.index.tolist()
    m_size = int(sg['n_i'].mode().iloc[0]) if not sg['n_i'].mode().empty else 1
    
    if chart_type == 'Xbar-R':
        sg['R'] = data.groupby(sg_col)[val_col].max() - data.groupby(sg_col)[val_col].min()
        Xbar_bar = sg['Xbar'].mean()
        R_bar = sg['R'].mean()
        A2 = unbiased_coefficient_fallback('A2', m_size)
        D3 = unbiased_coefficient_fallback('D3', m_size)
        D4 = unbiased_coefficient_fallback('D4', m_size)
        
        chart1 = pd.DataFrame({'point': sg['Xbar'], 'CL': Xbar_bar, 'LCL': Xbar_bar - A2 * R_bar, 'UCL': Xbar_bar + A2 * R_bar}, index=subgroup_indices)
        chart2 = pd.DataFrame({'point': sg['R'], 'CL': R_bar, 'LCL': D3 * R_bar, 'UCL': D4 * R_bar}, index=subgroup_indices)
        return chart1, chart2
    elif chart_type == 'Xbar-s':
        sg['s'] = data.groupby(sg_col)[val_col].std(ddof=1).fillna(0)
        Xbar_bar = sg['Xbar'].mean()
        s_bar = sg['s'].mean()
        A3 = unbiased_coefficient_fallback('A3', m_size)
        B3 = unbiased_coefficient_fallback('B3', m_size)
        B4 = unbiased_coefficient_fallback('B4', m_size)
        
        chart1 = pd.DataFrame({'point': sg['Xbar'], 'CL': Xbar_bar, 'LCL': Xbar_bar - A3 * s_bar, 'UCL': Xbar_bar + A3 * s_bar}, index=subgroup_indices)
        chart2 = pd.DataFrame({'point': sg['s'], 'CL': s_bar, 'LCL': B3 * s_bar, 'UCL': B4 * s_bar}, index=subgroup_indices)
        return chart1, chart2
    elif chart_type == 'I-MR':
        Xbar = data[val_col].mean()
        MR_i = data[val_col].rolling(window=window).apply(lambda x: x.max() - x.min())
        MR_bar = MR_i[window-1:].mean()
        D3 = unbiased_coefficient_fallback('D3', window)
        D4 = unbiased_coefficient_fallback('D4', window)
        d2 = unbiased_coefficient_fallback('d2', window)
        
        chart1 = pd.DataFrame({'point': data[val_col], 'CL': Xbar, 'LCL': Xbar - 3 * MR_bar / d2, 'UCL': Xbar + 3 * MR_bar / d2}, index=data.index)
        chart2 = pd.DataFrame({'point': MR_i, 'CL': MR_bar, 'LCL': D3 * MR_bar, 'UCL': D4 * MR_bar}, index=data.index)
        return chart1, chart2
    return None, None
