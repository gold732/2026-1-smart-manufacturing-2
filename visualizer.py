import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

def plot_boxplot_dashboard(df, x_col, y_col):
    """강의록 08_공정능력분석 12페이지의 박스플롯 옵션 재현 (points='all')"""
    fig = px.box(
        df, x=x_col, y=y_col, color=x_col, points='all',
        title=f'[{y_col}] 부품/라인별 박스플롯 분포 분석',
        template='seaborn'
    )
    fig.update_layout(width=800, height=450)
    return fig

def plot_qq_dashboard(values):
    """강의록 08_공정능력분석 13~14페이지 Q-Q Plot 직선 가이드 컴포넌트 재현"""
    z_value = stats.zscore(values)
    (x, y), _ = stats.probplot(z_value, dist='norm')
    
    fig = px.scatter(
        x=x, y=y, title='정규성 검증을 위한 Q-Q Plot',
        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
        template='seaborn'
    )
    fig.add_shape(type='line', x0=-3, y0=-3, x1=3, y1=3, line=dict(color='red', width=2))
    fig.update_layout(width=420, height=420)
    return fig

def plot_process_capability_histogram(df, val_col, lsl, usl, metrics):
    """강의록 08_공정능력분석 23~24페이지 한계 가이드가 동봉된 스펙 분석 가시화기"""
    fig = px.histogram(
        df, x=val_col, nbins=20, marginal="box",
        title=f'{val_col.capitalize()} 공정능력분석 히스토그램 종합 리포트',
        opacity=0.6, template='seaborn'
    )
    fig.add_vline(x=lsl, line_width=2, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=usl, line_width=2, line_dash="dash", line_color="red", annotation_text="USL")
    
    annotation_text = (
        f"<b>공정 지표 스코어</b><br>"
        f"Cp = {metrics['Cp']:.4f}<br>"
        f"Cpk = {metrics['Cpk']:.4f}<br>"
        f"Pp = {metrics['Pp']:.4f}<br>"
        f"Ppk = {metrics['Ppk']:.4f}"
    )
    fig.add_annotation(
        xref='paper', yref='paper', x=0.98, y=0.85,
        text=annotation_text, showarrow=False, align="left",
        bgcolor="white", bordercolor="black", borderwidth=1
    )
    fig.update_layout(width=850, height=500)
    return fig

def plot_control_chart_dashboard(chart1, chart2, chart_type='Xbar-R'):
    """강의록 09_통계적공정관리 12~14페이지 선조합 제어 스타일 규칙 완벽 수용"""
    names = chart_type.split('-')
    sub_title1 = names[0]
    sub_title2 = names[1] if len(names) > 1 else 'MR'
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'{sub_title1} 관리도', f'{sub_title2} 관리도'))
    
    # 1. 메인 통계 변동 차트 바인딩 (UCL/CL/LCL 점선 및 우측 끝 수치 라벨 추가)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['point'], mode='lines+markers', name=sub_title1, marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=1, col=1)
    
    # 2. 산포 가용 차트 바인딩
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['point'], mode='lines+markers', name=sub_title2, marker=dict(size=8, color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=2, col=1)
    
    # 강의록 13페이지 명시: 마지막 인덱스 우측 여백에 실시간 통계 경계 수치 문자열 Annotation 투영
    for i, c_data in enumerate([chart1, chart2], start=1):
        last_idx = c_data.index[-1]
        # 인덱스가 숫자인 경우와 문자열인 경우 구분하여 문자열 오프셋 마진 부여
        try:
            x_pos = last_idx + (len(c_data) * 0.05)
        except TypeError:
            x_pos = last_idx
            
        fig.add_annotation(x=x_pos, y=c_data['UCL'].iloc[-1], text=f"UCL={c_data['UCL'].iloc[-1]:.4f}", showarrow=False, font=dict(color='magenta'), row=i, col=1, xshift=15)
        fig.add_annotation(x=x_pos, y=c_data['CL'].iloc[-1], text=f"CL={c_data['CL'].iloc[-1]:.4f}", showarrow=False, font=dict(color='green'), row=i, col=1, xshift=15)
        fig.add_annotation(x=x_pos, y=c_data['LCL'].iloc[-1], text=f"LCL={c_data['LCL'].iloc[-1]:.4f}", showarrow=False, font=dict(color='red'), row=i, col=1, xshift=15)

    fig.update_layout(width=900, height=700, title_text=f"{chart_type} 통계적 모니터링 관리 시각화 대시보드", showlegend=False, template='seaborn')
    return fig
