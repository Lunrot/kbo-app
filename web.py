import os
import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# CSV 읽어오는 부분은 그대로 유지

def read_kbo_data(file_path="./2025.csv"):
    df = pd.read_csv(file_path)
    kia_index = df[df['날짜'] == 'KIA'].index[0]
    first_table = df.iloc[kia_index:kia_index + 10]
    date_columns = [col for col in first_table.columns if col != '날짜']

    last_col_indices = []
    for _, row in first_table.iterrows():
        for i in reversed(range(len(date_columns))):
            if pd.notna(row[date_columns[i]]):
                last_col_indices.append(i)
                break

    last_index = max(last_col_indices)
    subset_df = df.iloc[12:23].dropna(subset=['날짜'])
    dates = [col for col in subset_df.columns if col != '날짜'][:last_index + 1]

    data = {
        row['날짜']: [int(row[date]) for date in dates]
        for _, row in subset_df.iterrows()
    }
    df_numeric = pd.DataFrame(data, index=range(len(dates)))
    teams = df_numeric.columns.tolist()

    return dates, data, df_numeric, teams


def interpolate_data(df, n_interp=5):
    interp_steps = (len(df) - 1) * n_interp + 1
    x_interp = np.linspace(0, len(df) - 1, interp_steps)
    interp_df = pd.DataFrame(index=x_interp)
    for col in df.columns:
        interp_df[col] = np.interp(x_interp, df.index, df[col])
    return x_interp, interp_df, interp_steps


def get_team_colors():
    return {
        'KIA': '#ea0029', '삼성': '#074ca1', 'LG': '#c30452', '두산': '#1a1748',
        'KT': '#000000', 'SSG': '#ce0e2d', '롯데': '#041e42',
        '한화': '#fc4e00', 'NC': '#315288', '키움': '#570514'
    }


# 데이터 준비
dates, data, df, teams = read_kbo_data()
x_interp, interp_df, interp_steps = interpolate_data(df, n_interp=1)
colors = get_team_colors()

# 축 범위 계산
ymin = int(np.floor((interp_df.min().min() - 3) / 5.0) * 5)
ymax = int(np.ceil((interp_df.max().max() + 3) / 5.0) * 5)
xtick_indices = list(range(0, len(dates), 5))
xtick_labels = [dates[i] for i in xtick_indices]

# Dash 앱 설정
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("2025 KBO 팀 승패 마진 그래프"),
    dcc.Graph(id='graph', style={'height': '700px'}),
    dcc.Slider(id='frame-slider', min=0, max=interp_steps-1, value=0, step=1),
    dcc.Checklist(
        id='team-select',
        options=[{'label': t, 'value': t} for t in teams],
        value=[],
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),
    html.Button('Play', id='play-button', n_clicks=0),
    html.Button('Pause', id='pause-button', n_clicks=0),
    dcc.Interval(id='interval', interval=20, n_intervals=0, disabled=True)
])

@app.callback(
    Output('interval', 'disabled'),
    [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')]
)
def control_interval(play_clicks, pause_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return True
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return True if button_id == 'pause-button' else False

@app.callback(
    Output('frame-slider', 'value'),
    [Input('interval', 'n_intervals')],
    [State('frame-slider', 'value')]
)
def update_slider(n_intervals, current):
    return (current + 1) % interp_steps

@app.callback(
    Output('graph', 'figure'),
    [Input('frame-slider', 'value'), Input('team-select', 'value')]
)
def update_graph(frame, selected):
    current_x = x_interp[:frame+1]
    fig = go.Figure()

    for team in teams:
        y_vals = interp_df[team].iloc[:frame+1].values

        # 날짜 인덱스를 정수로 변환한 뒤 날짜 문자열로 매핑
        date_indices = np.round(current_x).astype(int)
        date_labels = [dates[i] if 0 <= i < len(dates) else "" for i in date_indices]

        opacity = 1.0 if not selected or team in selected else 0.2
        label_texts = [team if i == len(y_vals) - 1 else '' for i in range(len(y_vals))]

        fig.add_trace(go.Scatter(
            x=current_x,
            y=y_vals,
            mode='lines+text',
            name=team,
            line=dict(color=colors[team], width=1),
            marker=dict(size=4),
            text=label_texts,
            textposition='top right',
            textfont=dict(color=colors[team], size=10),
            opacity=opacity,
            customdata=np.array(date_labels).reshape(-1, 1),  # 날짜 문자열 전달
            hovertemplate=(
                f"<b>팀명:</b> {team}<br>" +
                "<b>날짜:</b> %{customdata[0]}<br>" +
                "<b>승패 마진:</b> %{y:.0f}<extra></extra>"
            )
        ))

    fig.update_yaxes(
        range=[ymin, ymax],
        tick0=ymin,
        dtick=5,
        title_text="승패"
    )
    fig.update_xaxes(
        range=[0, len(df) - 1 + 1.5],
        tickmode='array',
        tickvals=xtick_indices,
        ticktext=xtick_labels,
        title_text="날짜"
    )
    fig.update_layout(
        title="2025 KBO 팀 승패 마진 그래프",
        legend_title_text="팀",
        margin=dict(t=60, b=60),
        uirevision='static',
        transition=dic(duration=0),
    )
    return fig




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port)
