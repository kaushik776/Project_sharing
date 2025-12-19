import json
import plotly
import plotly.graph_objs as go
from flask import Flask, render_template, request
from utils.f1_data import predict_strategy, get_circuit_layout
from utils.f1_data import get_detailed_telemetry, TRACKS, DRIVERS, YEARS


app = Flask(__name__)


@app.route('/')
def home():
    """Renders the homepage of the PIT STOP application.

    Returns:
        str: The rendered HTML content of 'home.html'.
    """
    return render_template('home.html')


@app.route('/simulator', methods=['GET', 'POST'])
def simulator() -> str:
    """Handles the Strategy Simulator page logic.

    On GET: Renders the simulation input form.
    On POST: Processes user input to generate a race strategy prediction
    and a circuit map visualization.

    Returns:
        str: The rendered HTML content of 'simulator.html', including
             prediction results and map data.
    """
    result = None
    mapJSON = None
    error = None

    if request.method == 'POST':
        track = request.form.get('track')
        compound = request.form.get('compound')
        stops = request.form.get('stops')

        # 1. Fetch Circuit Map
        circuit_data = get_circuit_layout(track)
        if circuit_data:
            fig_map = go.Figure(go.Scatter(
                x=circuit_data['x'], y=circuit_data['y'],
                mode='lines', line=dict(color='#e10600', width=5),  # Red Track
                hoverinfo='skip'
            ))
            fig_map.update_layout(
                title=dict(text=f"{circuit_data['name']}",
                           font=dict(color='white', size=20)),
                template='plotly_dark',
                xaxis=dict(visible=False, fixedrange=True),
                yaxis=dict(visible=False, fixedrange=True),
                margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            mapJSON = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. Run Strategy ML (Only Metrics)
        data, err = predict_strategy(track, compound, stops)
        if err:
            error = err
        elif data:
            result = data

    return render_template('simulator.html',
                           tracks=TRACKS,
                           result=result,
                           mapJSON=mapJSON,
                           error=error)


@app.route('/telemetry', methods=['GET', 'POST'])
def telemetry() -> str:
    """Handles the Telemetry Dashboard page logic.

    On GET: Renders the telemetry selection form.
    On POST: Fetches and processes telemetry data (Speed Trace & Race Pace)
             for the selected drivers and session.

    Returns:
        str: The rendered HTML content of 'telemetry.html', including charts
             and race statistics if data is available.
    """
    data = None
    paceJSON = None
    telemetryJSON = None
    error = None

    if request.method == 'POST':
        year = request.form.get('year')
        race = request.form.get('race')
        d1 = request.form.get('driver1')
        d2 = request.form.get('driver2')

        result, err = get_detailed_telemetry(year, race, d1, d2)

        if err:
            error = err
        elif result:
            data = result

            # CHART 1: Speed Telemetry
            fig_tel = go.Figure()
            colors = {d1: '#3671C6', d2: '#F91536'}

            for item in result['telemetry_data']:
                drv = item['driver']
                fig_tel.add_trace(go.Scatter(
                    x=item['distance'], y=item['speed'], mode='lines',
                    name=f"{drv}",
                    line=dict(color=colors.get(drv, 'white'), width=2)
                ))

            fig_tel.update_layout(
                title='Fastest Lap Telemetry (Speed Trace)',
                xaxis_title='Distance (m)', yaxis_title='Speed (km/h)',
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(orientation="h", y=1.1)
            )
            telemetryJSON = json.dumps(fig_tel,
                                       cls=plotly.utils.PlotlyJSONEncoder)

            # CHART 2: Race Pace
            fig_pace = go.Figure()
            for item in result['pace_data']:
                drv = item['driver']
                fig_pace.add_trace(go.Scatter(
                    x=item['x'], y=item['y'], mode='markers', name=drv,
                    marker=dict(size=8,
                                color=colors.get(drv, 'white'),
                                opacity=0.8)
                ))

            fig_pace.update_layout(
                title='Driver Comparison (Race Pace)',
                xaxis_title='Lap Number', yaxis_title='Lap Time (s)',
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(orientation="h", y=1.1)
            )
            paceJSON = json.dumps(fig_pace, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('telemetry.html',
                           years=YEARS, tracks=TRACKS, drivers=DRIVERS,
                           data=data, paceJSON=paceJSON,
                           telemetryJSON=telemetryJSON, error=error)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
