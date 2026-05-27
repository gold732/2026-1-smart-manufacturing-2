import numpy as np
import pandas as pd
from scipy.stats import shapiro, boxcox, norm
from scipy.special import gamma

# ==========================================
# 1. 통계적 불편화 상수 보정 모듈
# ==========================================

def calc_unbiased_const(const_name, n):
    """
    표본 크기에 따른 품질관리용 불편화 상수를 계산하거나 테이블에서 참조합니다.
    """
    try:
        d_table = pd.read_csv('./data/unbiased_capability_analysis.csv')
    except FileNotFoundError:
        try:
            d_table = pd.read_csv('../data/smart/unbiased_capability_analysis.csv')
        except FileNotFoundError:
            # 파일 누락 시를 대비한 수학적 공식 기반 Fallback 처리
            if const_name == 'c4':
                if n <= 1: return 1.0
                return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
            elif const_name == 'd2':
                if n < 51:
                    d2_dict = {2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534, 7:2.704, 8:2.847, 9:2.970, 10:3.078}
                    return d2_dict.get(n, 3.4873 + 0.0250141 * n - 0.00009823 * n**2)
                return 3.4873 + 0.0250141 * n - 0.00009823 * n**2
            return None

    n = int(n)
    if const_name == 'd2':
        if n < 51:
            return d_table['d2'][n - 2]
        return 3.4873 + 0.0250141 * n - 0.00009823 * n**2
    elif const_name == 'd3':
        if n < 26:
            return d_table['d3'][n - 2]
        return 0.80818 - 0.051871 * n + 0.00005096 * n**2 - 0.00000019 * n**3
    elif const_name == 'd4':
        if n < 26:
            return d_table['d4'][n - 2]
        return 2.88606 + 0.051313 * n - 0.00049243 * n**2 + 0.00000188 * n**3
    elif const_name == 'c2':
        return (np.sqrt(2) / np.sqrt(n)) * (gamma(n / 2) / gamma((n - 1) / 2))
    elif const_name == 'c3':
        c2 = (np.sqrt(2) / np.sqrt(n)) * (gamma(n / 2) / gamma((n - 1) / 2))
        return np.sqrt((n - 1) / n - c2**2)
    elif const_name == 'c4':
        return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
    return None


def unbiased_coefficient(coef_name, m):
    """
    관리도 한계선 산출을 위한 관리도용 계표 상수를 매핑합니다.
    """
    try:
        d_table = pd.read_csv('./data/unbiased_control_chart.csv')
    except FileNotFoundError:
        try:
            d_table = pd.read_csv('../data/smart/unbiased_control_chart.csv')
        except FileNotFoundError:
            # 안전한 연산을 위한 백업 테이블
            fallback_chart = {
                'A2': {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483},
                'D3': {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136},
                'D4': {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004}
            }
            if coef_name in fallback_chart:
                return fallback_chart[coef_name].get(m, 1.0 if 'D4' in coef_name else 0.0)
            return 1.0

    m = int(m)
    if (coef_name in d_table.columns[1:]) and m >= 2 and m <= 25:
        return d_table[coef_name][m - 2]
    else:
        return d_table[coef_name].iloc[-1]


# ==========================================
# 2. 정규성 검정 및 변환 모듈
# ==========================================

def normality_test_and_transform(data_series, lsl_orig=None, usl_orig=None):
    """
    Shapiro-Wilk 정규성 검정을 수행하며, 기각 시 Box-Cox 변환 및 규격 한계를 동시 변환합니다.
    """
    stat, p = shapiro(data_series)
    
    if p >= 0.05:
        return data_series, lsl_orig, usl_orig, p, False, 1.0
    else:
        # 음수 데이터 포함 시 척도 시프트 처리
        if (data_series <= 0).any():
            shift_val = abs(data_series.min()) + 1.0
            data_to_transform = data_series + shift_val
            if lsl_orig is not None: lsl_orig += shift_val
            if usl_orig is not None: usl_orig += shift_val
        else:
            data_to_transform = data_series

        transformed_data, lambda_val = boxcox(data_to_transform)
        
        # 데이터 스케일 변환에 따른 LSL / USL 동시 변환 로직
        lsl_trans = lsl_orig
        usl_trans = usl_orig
        
        if lambda_val == 0:
            if lsl_orig is not None and lsl_orig > 0: lsl_trans = np.log(lsl_orig)
            if usl_orig is not None and usl_orig > 0: usl_trans = np.log(usl_orig)
        else:
            if lsl_orig is not None and lsl_orig > 0: lsl_trans = (lsl_orig**lambda_val - 1) / lambda_val
            if usl_orig is not None and usl_orig > 0: usl_trans = (usl_orig**lambda_val - 1) / lambda_val
            
        _, p_trans = shapiro(transformed_data)
        return pd.Series(transformed_data, index=data_series.index), lsl_trans, usl_trans, p_trans, True, lambda_val


# ==========================================
# 3. 공정능력분석(Process Capability) 모듈
# ==========================================

def process_capability(data, sg_name, var_name, LSL, USL):
    """
    단기 변동(군내) 및 장기 변동(전체) 표준편차를 분리하여 공정능력지수를 계산합니다.
    """
    mean_sg = data.groupby(sg_name)[var_name].mean()
    sigma_sg = data.groupby(sg_name)[var_name].std()
    
    n_i_counts = data[sg_name].value_counts()
    m_mode = n_i_counts.mode()
    m = int(m_mode.iloc[0]) if not m_mode.empty else int(len(data) / len(mean_sg))
    
    x_bar = data[var_name].mean()
    sigma_hat = data[var_name].std(ddof=1)
    
    c4_overall = calc_unbiased_const('c4', len(data))
    sigma_overall = sigma_hat / c4_overall if c4_overall else sigma_hat
    
    # 합동 표준편차 산출 및 불편화 상수를 통한 군내 표준편차 추정
    sigma_p = np.sqrt(np.sum(sigma_sg**2) / len(sigma_sg))
    subgroup_delta_d = len(data) - len(sigma_sg) + 1
    c4_within = calc_unbiased_const('c4', subgroup_delta_d)
    sigma_within = sigma_p / c4_within if c4_within else sigma_p
    
    # Cp, Cpk 및 Pp, Ppk 지수 산출
    Cp = (USL - LSL) / (6 * sigma_within) if LSL is not None and USL is not None else np.nan
    Cpk = min((USL - x_bar) / (3 * sigma_within), (x_bar - LSL) / (3 * sigma_within)) if LSL is not None and USL is not None else np.nan
    
    Pp = (USL - LSL) / (6 * sigma_overall) if LSL is not None and USL is not None else np.nan
    Ppk = min((USL - x_bar) / (3 * sigma_overall), (x_bar - LSL) / (3 * sigma_overall)) if LSL is not None and USL is not None else np.nan
    
    return Cp, Cpk, Pp, Ppk, x_bar, sigma_within, sigma_overall


# ==========================================
# 4. 계량형 관리도(Variable Control Chart) 모듈
# ==========================================

def generate_value_chart(data, sg_name, var_name, chart_type='Xbar-R', window=3):
    """
    Xbar-R, Xbar-s, I-MR 관리도의 중심선 및 통계적 관리한계선을 생성합니다.
    """
    if chart_type == 'Xbar-R':
        sg = pd.DataFrame()
        sg['Xbar'] = data.groupby(sg_name)[var_name].mean()
        sg['R'] = data.groupby(sg_name)[var_name].max() - data.groupby(sg_name)[var_name].min()
        sg['n_i'] = data[sg_name].value_counts()
        
        Xbar_bar = sg['Xbar'].mean()
        R_bar = sg['R'].mean()
        m_val = sg['n_i'].mode().iloc[0] if not sg['n_i'].mode().empty else 5
        
        A2 = unbiased_coefficient('A2', m_val)
        D3 = unbiased_coefficient('D3', m_val)
        D4 = unbiased_coefficient('D4', m_val)
        
        Xbar_chart = pd.DataFrame(index=sg.index)
        Xbar_chart['point'] = sg['Xbar']
        Xbar_chart['CL'] = Xbar_bar
        Xbar_chart['LCL'] = Xbar_bar - A2 * R_bar
        Xbar_chart['UCL'] = Xbar_bar + A2 * R_bar
        
        R_chart = pd.DataFrame(index=sg.index)
        R_chart['point'] = sg['R']
        R_chart['CL'] = R_bar
        R_chart['LCL'] = D3 * R_bar
        R_chart['UCL'] = D4 * R_bar
        
        return Xbar_chart, R_chart

    elif chart_type == 'Xbar-s':
        sg = pd.DataFrame()
        sg['Xbar'] = data.groupby(sg_name)[var_name].mean()
        sg['s'] = data.groupby(sg_name)[var_name].std(ddof=1).fillna(0)
        sg['n_i'] = data[sg_name].value_counts()
        
        Xbar_bar = sg['Xbar'].mean()
        s_bar = sg['s'].mean()
        m_val = sg['n_i'].mode().iloc[0] if not sg['n_i'].mode().empty else 5
        
        A3 = unbiased_coefficient('A3', m_val)
        B3 = unbiased_coefficient('B3', m_val)
        B4 = unbiased_coefficient('B4', m_val)
        
        Xbar_chart = pd.DataFrame(index=sg.index)
        Xbar_chart['point'] = sg['Xbar']
        Xbar_chart['CL'] = Xbar_bar
        Xbar_chart['LCL'] = Xbar_bar - A3 * s_bar
        Xbar_chart['UCL'] = Xbar_bar + A3 * s_bar
        
        s_chart = pd.DataFrame(index=sg.index)
        s_chart['point'] = sg['s']
        s_chart['CL'] = s_bar
        s_chart['LCL'] = B3 * s_bar
        s_chart['UCL'] = B4 * s_bar
        
        return Xbar_chart, s_chart

    elif chart_type == 'I-MR':
        sg = data.set_index(sg_name).sort_index()
        w = window
        
        Xbar = sg[var_name].mean()
        MR_i = sg[var_name].rolling(window=w).apply(lambda x: x.max() - x.min(), raw=True)
        MR_bar = MR_i.iloc[w-1:].mean()
        
        D3 = unbiased_coefficient('D3', w)
        D4 = unbiased_coefficient('D4', w)
        d2 = unbiased_coefficient('d2', w)
        if d2 == 0 or d2 is None: d2 = 1.128
        
        I_chart = pd.DataFrame(index=sg.index)
        I_chart['point'] = sg[var_name]
        I_chart['CL'] = Xbar
        I_chart['LCL'] = Xbar - 3 * MR_bar / d2
        I_chart['UCL'] = Xbar + 3 * MR_bar / d2
        
        MR_chart = pd.DataFrame(index=sg.index)
        MR_chart['point'] = MR_i
        MR_chart['CL'] = MR_bar
        MR_chart['LCL'] = D3 * MR_bar
        MR_chart['UCL'] = D4 * MR_bar
        
        return I_chart, MR_chart
        
    return None


# ==========================================
# 5. 계수형 관리도(Attribute Control Chart) 모듈
# ==========================================

def generate_count_chart(df_raw, sg_name, n_i, var_name, chart_type='NP'):
    """
    NP, P, C, U 관리도의 관리한계선을 계산하며, 표본 크기 가변성에 대응합니다.
    """
    data = df_raw.set_index(sg_name)
    
    if chart_type == 'NP':
        np_bar = data[var_name].sum() / len(data)
        p_bar = data[var_name].sum() / data[n_i].sum()
        
        NP_chart = pd.DataFrame(index=data.index)
        NP_chart['point'] = data[var_name]
        NP_chart['CL'] = np_bar
        NP_chart['LCL'] = np_bar - 3 * np.sqrt(np_bar * (1 - p_bar))
        NP_chart['UCL'] = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
        
        NP_chart['LCL'] = NP_chart['LCL'].clip(lower=0)
        return NP_chart

    elif chart_type == 'P':
        p_bar = data[var_name].sum() / data[n_i].sum()
        
        P_chart = pd.DataFrame(index=data.index)
        P_chart['point'] = data[var_name] / data[n_i]
        P_chart['CL'] = p_bar
        
        # 가변 표본 크기 시리즈(data[n_i])에 대응하는 동적 UCL/LCL 한계선 연산 적용
        P_chart['LCL'] = p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / data[n_i])
        P_chart['UCL'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / data[n_i])
        
        P_chart['LCL'] = P_chart['LCL'].clip(lower=0)
        return P_chart

    elif chart_type == 'C':
        c_bar = data[var_name].mean()
        
        C_chart = pd.DataFrame(index=data.index)
        C_chart['point'] = data[var_name]
        C_chart['CL'] = c_bar
        C_chart['LCL'] = c_bar - 3 * np.sqrt(c_bar)
        C_chart['UCL'] = c_bar + 3 * np.sqrt(c_bar)
        
        C_chart['LCL'] = C_chart['LCL'].clip(lower=0)
        return C_chart

    elif chart_type == 'U':
        u_bar = data[var_name].sum() / data[n_i].sum()
        
        U_chart = pd.DataFrame(index=data.index)
        U_chart['point'] = data[var_name] / data[n_i]
        U_chart['CL'] = u_bar
        
        # 가변 표본 크기 시리즈(data[n_i])에 대응하는 단위당 결함률 UCL/LCL 동적 공식 연산 적용
        U_chart['LCL'] = u_bar - 3 * np.sqrt(u_bar / data[n_i])
        U_chart['UCL'] = u_bar + 3 * np.sqrt(u_bar / data[n_i])
        
        U_chart['LCL'] = U_chart['LCL'].clip(lower=0)
        return U_chart
        
    return None
