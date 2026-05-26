import numpy as np
import pandas as pd
from scipy.stats import shapiro, boxcox
from scipy.special import gamma

def calc_unbiased_const(const_name, n):
    """강의록 08_공정능력분석 15~16페이지의 수학적 근사식 및 불편화 상수 완벽 보존"""
    if n <= 1:
        return 1.0
    
    if const_name == 'd2':
        if n < 51:
            # d2 테이블 대용 Fallback용 기본 매핑 데이터셋 호출 시도
            try:
                d_table = pd.read_csv('unbiased_control_chart.csv')
                return float(d_table['d2'].iloc[int(n)-2])
            except Exception:
                return 3.4873 + 0.0250141 * n - 0.00009823 * (n ** 2)
        return 3.4873 + 0.0250141 * n - 0.00009823 * (n ** 2)
        
    elif const_name == 'd3':
        if n < 26:
            try:
                d_table = pd.read_csv('unbiased_control_chart.csv')
                return float(d_table['d3'].iloc[int(n)-2])
            except Exception:
                pass
        return 0.80818 - 0.051871 * n + 0.00005096 * (n ** 2) - 0.00000019 * (n ** 3)
        
    elif const_name == 'd4':
        if n < 26:
            try:
                d_table = pd.read_csv('unbiased_control_chart.csv')
                return float(d_table['d4'].iloc[int(n)-2])
            except Exception:
                pass
        return 2.88606 + 0.051313 * n - 0.00049243 * (n ** 2) + 0.00000188 * (n ** 3)
        
    elif const_name == 'c2':
        return (np.sqrt(2) / np.sqrt(n)) * (gamma(n / 2) / gamma((n - 1) / 2))
        
    elif const_name == 'c3':
        c2 = (np.sqrt(2) / np.sqrt(n)) * (gamma(n / 2) / gamma((n - 1) / 2))
        return np.sqrt((n - 1) / n - c2 ** 2)
        
    elif const_name == 'c4':
        return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
        
    return 1.0

def unbiased_coefficient_fallback(coef_name, m):
    """강의록 09_통계적공정관리 4~5페이지 제어 차트 상수 테이블 완벽 보존"""
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
    """Shapiro-Wilk 정규성 판정 공식"""
    stat, p = shapiro(values)
    return p, p >= 0.05

def analyze_process_capability(df, sg_col, val_col, lsl, usl):
    """강의록 08_공정능력분석 16~17페이지 장/단기 변동 분석 모델 수식 구현"""
    mean_sg = df.groupby(sg_col)[val_col].mean()
    sigma_sg = df.groupby(sg_col)[val_col].std().fillna(0)
    
    x_bar = df[val_col].mean()
    sigma_hat = df[val_col].std(ddof=1)
    
    # 1. Overall 변동 지표 (장기 표준편차 및 Pp/Ppk 수식 완벽 보존)
    c4_overall = calc_unbiased_const('c4', len(df))
    sigma_overall = sigma_hat / c4_overall if c4_overall > 0 else sigma_hat
    
    # 2. Within 변동 지표 (단기 합동표준편차 분기식 수식 완벽 보존)
    sigma_p = np.sqrt(np.sum(sigma_sg**2) / len(sigma_sg))
    subgroup_sizes = df[sg_col].value_counts()
    m_size = int(subgroup_sizes.mode().iloc[0]) if not subgroup_sizes.mode().empty else 2
    
    c4_within = calc_unbiased_const('c4', len(df) - len(sigma_sg) + 1)
    sigma_within = sigma_p / c4_within if c4_within > 0 else sigma_p
    
    # 공정지수 정량식 산출
    Cp = (usl - lsl) / (6 * sigma_within) if sigma_within > 0 else 0
    Cpk = min((usl - x_bar) / (3 * sigma_within), (x_bar - lsl) / (3 * sigma_within)) if sigma_within > 0 else 0
    
    Pp = (usl - lsl) / (6 * sigma_overall) if sigma_overall > 0 else 0
    Ppk = min((usl - x_bar) / (3 * sigma_overall), (x_bar - lsl) / (3 * sigma_overall)) if sigma_overall > 0 else 0
    
    return {
        "x_bar": x_bar, "sigma_overall": sigma_overall, "sigma_within": sigma_within,
        "Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk
    }

def apply_box_cox(values, lsl, usl):
    """비정규 분포 타파용 Box-Cox 데이터 스케일 아웃 및 상하한 동시 변환 로직"""
    transformed_vals, lambda_val = boxcox(values)
    
    def transform_value(x, lmbda):
        if lmbda == 0:
            return np.log(x)
        return (x**lmbda - 1) / lmbda

    lsl_t = transform_value(lsl, lambda_val)
    usl_t = transform_value(usl, lambda_val)
    return transformed_vals, lsl_t, usl_t, lambda_val

def generate_value_chart_data(data, sg_col, val_col, chart_type='Xbar-R', window=3):
    """강의록 09_통계적공정관리 5~7페이지 계량형 제어 상하한 한계 계산식 최빈값 기준 적용"""
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
