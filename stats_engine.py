import numpy as np
import pandas as pd
from scipy.stats import shapiro, boxcox, norm
from scipy.special import gamma

# ==========================================
# 1. 불편화 상수 및 계수 계산 엔진 파트
# ==========================================

def calc_unbiased_const(const_name, n):
    """
    강의록 15-16페이지 불편화 상수 계산 기준 구현
    """
    try:
        # 파일 경로 가독성 및 이식성을 고려한 상대 경로 지정
        d_table = pd.read_csv('./data/unbiased_capability_analysis.csv')
    except FileNotFoundError:
        try:
            d_table = pd.read_csv('../data/smart/unbiased_capability_analysis.csv')
        except FileNotFoundError:
            # 파일이 없을 경우 강의록 d2, c4 공식 및 근사치 기반 Fallback 처리
            if const_name == 'c4':
                if n <= 1: return 1.0
                return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
            elif const_name == 'd2':
                if n < 51:
                    # 마이그레이션 대비 하드코딩 백업 (n=2~10 기준 대표값)
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
    강의록 4-5페이지 관리도용 불편화 계수 매핑
    """
    try:
        d_table = pd.read_csv('./data/unbiased_control_chart.csv')
    except FileNotFoundError:
        try:
            d_table = pd.read_csv('../data/smart/unbiased_control_chart.csv')
        except FileNotFoundError:
            # Fallback 매핑 테이블 (가장 빈번한 m=4, 5 기준 예외 방지)
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
# 2. 정규성 검정 및 정규화(Box-Cox) 엔진 파트
# ==========================================

def normality_test_and_transform(data_series, lsl_orig=None, usl_orig=None):
    """
    정규성 검정을 수행하고, 불만족 시 Box-Cox 변환 및 규격 한계(LSL/USL)를 동시 변환하여 반환함.
    강의록 27-29페이지의 규격 동시 변환 의무 조건 반영 지침 충족.
    """
    stat, p = shapiro(data_series)
    
    if p >= 0.05:
        # 정규성 만족 시 원본 데이터와 원본 규격 유지
        return data_series, lsl_orig, usl_orig, p, False, 1.0
    else:
        # 정규성 불만족 시 Box-Cox 거듭제곱 변환 수행
        # 변환 데이터 양수 조건 확보를 위해 최솟값 체크 및 보정
        if (data_series <= 0).any():
            shift_val = abs(data_series.min()) + 1.0
            data_to_transform = data_series + shift_val
            if lsl_orig is not None: lsl_orig += shift_val
            if usl_orig is not None: usl_orig += shift_val
        else:
            data_to_transform = data_series

        transformed_data, lambda_val = boxcox(data_to_transform)
        
        # [CRITICAL 수정을 해야만 하는 부분]: 규격 데이터도 변환 척도(Scale)로 동시 변환 계산 처리
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
# 3. 공정능력분석(Capability Analysis) 엔진 파트
# ==========================================

def process_capability(data, sg_name, var_name, LSL, USL):
    """
    강의록 16-17페이지 공정능력지수 공식 적용 단기/장기 분리 정밀 연산 함수
    """
    # 각 부분군별 평균 및 표준편차 산출
    mean_sg = data.groupby(sg_name)[var_name].mean()
    sigma_sg = data.groupby(sg_name)[var_name].std()
    
    # 부분군 표본 크기 최빈값 연산 처리 및 예외 처리
    n_i_counts = data[sg_name].value_counts()
    m_mode = n_i_counts.mode()
    m = int(m_mode.iloc[0]) if not m_mode.empty else int(len(data) / len(mean_sg))
    
    # 전체 장기 지표 계산용
    x_bar = data[var_name].mean()
    sigma_hat = data[var_name].std(ddof=1)
    
    c4_overall = calc_unbiased_const('c4', len(data))
    sigma_overall = sigma_hat / c4_overall if c4_overall else sigma_hat
    
    # 군내 변동(단기 단기 표준편차) 계산 공식 구현
    sigma_p = np.sqrt(np.sum(sigma_sg**2) / len(sigma_sg))
    
    # 합동 표준편차 불편화 상수를 통한 보정 연산 수행
    subgroup_delta_d = len(data) - len(sigma_sg) + 1
    c4_within = calc_unbiased_const('c4', subgroup_delta_d)
    sigma_within = sigma_p / c4_within if c4_within else sigma_p
    
    # 단기 및 장기 공정능력지수 최종 산출 연산
    Cp = (USL - LSL) / (6 * sigma_within) if LSL is not None and USL is not None else np.nan
    Cpk = min((USL - x_bar) / (3 * sigma_within), (x_bar - LSL) / (3 * sigma_within)) if LSL is not None and USL is not None else np.nan
    
    Pp = (USL - LSL) / (6 * sigma_overall) if LSL is not None and USL is not None else np.nan
    Ppk = min((USL - x_bar) / (3 * sigma_overall), (x_bar - LSL) / (3 * sigma_overall)) if LSL is not None and USL is not None else np.nan
    
    return Cp, Cpk, Pp, Ppk, x_bar, sigma_within, sigma_overall


# ==========================================
# 4. 통계적 공정관리(SPC 계량형) 엔진 파트
# ==========================================

def generate_value_chart(data, sg_name, var_name, chart_type='Xbar-R', window=3):
    """
    강의록 5-7페이지 계량형 관리도 기하 통계 한계선 계산 함수
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
        if d2 == 0 or d2 is None: d2 = 1.128  # 기본 윈도우 스케일링 세이프티
        
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
# 5. 통계적 공정관리(SPC 계수형) 엔진 파트
# ==========================================

def generate_count_chart(df_raw, sg_name, n_i, var_name, chart_type='NP'):
    """
    강의록 9-10페이지 계수형 불량/결함 공정관리도 연산 매커니즘 구현 함수
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
        
        # 하한선 음수 치환 방지 보정 처리 적용
        NP_chart['LCL'] = NP_chart['LCL'].clip(lower=0)
        return NP_chart

    elif chart_type == 'P':
        p_bar = data[var_name].sum() / data[n_i].sum()
        
        P_chart = pd.DataFrame(index=data.index)
        P_chart['point'] = data[var_name] / data[n_i]
        P_chart['CL'] = p_bar
        
        # [CRITICAL 수정을 해야만 하는 부분]: 표본 가변성 변동 고려 수식(data[n_i] 시리즈 연산) 동적 바인딩 적용
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
        
        # [CRITICAL 수정을 해야만 하는 부분]: 표본 크기 가변성에 맞춘 단위당 결함률 UCL/LCL 동적 공식 매핑 수정
        U_chart['LCL'] = u_bar - 3 * np.sqrt(u_bar / data[n_i])
        U_chart['UCL'] = u_bar + 3 * np.sqrt(u_bar / data[n_i])
        
        U_chart['LCL'] = U_chart['LCL'].clip(lower=0)
        return U_chart
        
    return None
