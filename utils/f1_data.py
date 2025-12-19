"""
F1 Data Analysis and Strategy Utilities.

This module provides a suite of tools for extracting, analyzing,
and predicting Formula 1 race data using the FastF1 library. It
handles session management, telemetry extraction, and machine
learning-based strategy predictions.

Key Features:
    - Safe Session Loading: Robust wrappers around FastF1 with error handling.
    - Circuit Mapping: Generates track layout coordinates from telemetry.
    - Strategy Prediction: Uses Linear Regression to estimate tyre degradation
      and race pace.
    - Comparative Telemetry: detailed side-by-side analysis of driver pace and
      speed traces.

Dependencies:
    - fastf1: For retrieving official timing and telemetry data.
    - sklearn: For linear regression models used in strategy prediction.
    - numpy: For numerical operations on lap data.

"""
import os
import numpy as np
import fastf1
from sklearn.linear_model import LinearRegression


CACHE_DIR = os.environ.get('FASTF1_CACHE_DIR', 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)


TRACKS: list[str] = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami', 'Monaco',
    'Spain', 'Canada', 'Austria', 'Great Britain', 'Hungary', 'Belgium',
    'Netherlands', 'Italy', 'Singapore', 'Japan', 'Qatar', 'USA', 'Mexico',
    'Brazil', 'Las Vegas', 'Abu Dhabi'
]

DRIVERS: list[str] = [
    'VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'NOR', 'PIA', 'ALO', 'STR',
    'GAS', 'OCO', 'ALB', 'SAR', 'TSU', 'RIC', 'BOT', 'ZHO', 'HUL', 'MAG'
]

YEARS: list[int] = [2024, 2023, 2022, 2021]


def get_session_safe(year: int | str, track: str, session_type='R') -> None:
    """Loads a FastF1 session object with built-in error handling.

    Args:
        year (int or str): The championship year (e.g., 2023).
        track (str): The name of the circuit or Grand Prix.
        session_type (str, optional): The type of session to load.
                                      Defaults to 'R' (Race).

    Returns:
        fastf1.core.Session | None: The loaded session object if successful,
                                    or None if an error occurs.
    """
    try:
        session = fastf1.get_session(int(year), track, session_type)
        session.load()
        return session
    except (ValueError, IndexError, KeyError) as e:
        print(f"Session load failed for {year} {track}: {e}")
        return None


def get_circuit_layout(track: str) -> dict | None:
    """Retrieves the coordinates of the circuit based on the fastest
       qualifying lap of 2023.

    Args:
        track (str): The name of the circuit.

    Returns:
        dict | None: A dictionary containing the track layout data with keys:
            - 'x' (list): List of X coordinates for the track path.
            - 'y' (list): List of Y coordinates for the track path.
            - 'name' (str): The official name of the event.
            Returns None if the session isn't be loaded or data extraction
            fails.
    """
    session = get_session_safe(2023, track, 'Q')
    if not session:
        return None

    try:
        lap = session.laps.pick_fastest()
        telemetry = lap.get_telemetry()

        return {
            'x': telemetry['X'].tolist(),
            'y': telemetry['Y'].tolist(),
            'name': session.event.EventName
        }
    except (ValueError, KeyError, IndexError) as e:
        print(f"Layout extraction failed for {track}: {e}")
        return None


def predict_strategy(track: str, compound: str,
                     stops: int | str) -> tuple | None:
    """Predicts overall race time and tyre degradation based on previous data.

    Args:
        track (str): The name of the Grand Prix circuit.
        compound (str): The tyre compound selected ('SOFT', 'MEDIUM', 'HARD').
        stops (int or str): The number of pit stops planned (1 or 2).

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary with keys 'total_time_min', 'degradation',
                    and 'stop_recommendation'.
            - None: (If successful) used for error handling consistency.

        OR (if an error occurs):
            - None: Indicates failure.
            - str: A description of the error.
    """
    # Use 2023 race data as the training set for the degradation model
    session = get_session_safe(2023, track, 'R')
    if not session:
        return None, "Historical data unavailable for this track."

    try:
        # Pre-process: Filter anomalies and non-representative laps (SC/VSC)
        laps = session.laps.pick_quicklaps().reset_index(drop=True)
        laps = laps.dropna(subset=['LapNumber', 'LapTime'])

        # Fit linear model-> to determine degradation slope
        X = laps['LapNumber'].values.reshape(-1, 1)
        y = laps['LapTime'].dt.total_seconds().values

        model = LinearRegression()
        model.fit(X, y)

        # Extrapolate pace for a standard 57-lap race distance
        future_laps = np.arange(1, 58).reshape(-1, 1)
        base_pace = model.predict(future_laps)

        stops = int(stops)
        stop_penalty = 22.0  # Average pit loss constant

        race_time_total = 0
        stop_laps = []

        # Standardize pit windows based on stop count
        if stops == 1:
            stop_laps = [25]
        elif stops == 2:
            stop_laps = [18, 38]

        # Aggregate lap times with compound deltas and pit penalties
        for i, time in enumerate(base_pace):
            lap_num = i + 1
            adjusted_time = time

            if compound == 'SOFT':
                adjusted_time -= 0.5
            elif compound == 'HARD':
                adjusted_time += 0.5

            if lap_num in stop_laps:
                adjusted_time += stop_penalty

            race_time_total += adjusted_time

        stop_str = "No Stops"
        if stops > 0:
            stop_str = ", ".join([f"Lap {x}" for x in stop_laps])

        return {
            'total_time_min': round(race_time_total / 60, 2),
            'degradation': round(model.coef_[0], 4),
            'stop_recommendation': stop_str
        }, None

    except (ValueError, IndexError, KeyError) as e:
        return None, str(e)


def get_detailed_telemetry(year: int, race: str, d1: str, d2: str
                           ) -> tuple | None:
    """Retrieves comparative telemetry data for two drivers in a specific race.

    This function fetches the race session and extracts two main datasets:
    1. Race Pace: Lap-by-lap comparison of lap times.
    2. Speed Trace: High-frequency speed vs distance telemetry for the fastest
                    lap of each driver.

    Args:
        year (int | str): The championship year.
        race (str): The name of the circuit or Grand Prix.
        d1 (str): Three-letter abbreviation for the first driver.
        d2 (str): Three-letter abbreviation for the second driver.

    Returns:
        tuple: A tuple containing (data, error).
            - data (dict | None): A dictionary with keys:
                - 'race_name' (str): Official event name.
                - 'pace_data' (list): List of dicts with 'driver', 'x', 'y'.
                - 'telemetry_data' (list): List of dicts with speed/distance
                                           traces for fastest laps.
                - 'winner_info' (dict): Metadata about the race winner.
            - error (str | None): Error message if data extraction fails,
                                  otherwise None.
    """
    session = get_session_safe(year, race, 'R')
    if not session:
        return None, f"Session data not found for {race} {year}"

    try:
        laps_d1 = session.laps.pick_driver(d1).pick_quicklaps()
        laps_d2 = session.laps.pick_driver(d2).pick_quicklaps()

        if laps_d1.empty or laps_d2.empty:
            return None, "Driver data unavailable."

        # Pace data for frontend visualization
        pace_data = []
        for d, laps in [(d1, laps_d1), (d2, laps_d2)]:
            df = laps[['LapNumber', 'LapTime']].copy()
            df = df.dropna()
            df['LapTime'] = df['LapTime'].dt.total_seconds()
            pace_data.append({
                'driver': d,
                'x': df['LapNumber'].tolist(),
                'y': df['LapTime'].tolist()
            })

        # Extract telemetry from the fastest lap for speed trace comparison
        fastest_d1 = laps_d1.pick_fastest()
        fastest_d2 = laps_d2.pick_fastest()

        telemetry_data = []

        def extract_tel(lap, driver_code: str) -> dict | None:
            """To extract distance, speed, and lap time from a lap object.

            Args:
                lap (fastf1.core.Lap): The lap object to process.
                driver_code(str): The driver's abbreviation to tag the
                                  data with.

            Returns:
                dict | None: A dictionary containing 'distance', 'speed', and
                             'lap_time' lists for plotting, or None if
                              extraction fails.
            """
            try:
                tel = lap.get_telemetry()
                return {
                    'driver': driver_code,
                    'distance': tel['Distance'].tolist(),
                    'speed': tel['Speed'].tolist(),
                    'lap_time': str(lap['LapTime']).split(' days ')[-1][0:10]
                }
            except (ValueError, IndexError, KeyError):
                return None

        t1 = extract_tel(fastest_d1, d1)
        t2 = extract_tel(fastest_d2, d2)
        if t1:
            telemetry_data.append(t1)
        if t2:
            telemetry_data.append(t2)

        # Retrieve race winner metadata
        try:
            winner_df = session.results.loc[session.results['Position'] == 1.0]
            if winner_df.empty:
                raise IndexError("No driver found with Position 1.0")

            winner_row = winner_df.iloc[0]
            winner_info = {
                'name': winner_row['Abbreviation'],
                'team': winner_row['TeamName'],
                'time': str(winner_row['Time']).split(' days ')[-1]
            }
        except (IndexError, KeyError, ValueError) as e:
            print(f"Winner stats extraction failed: {e}")
            winner_info = {'name': 'N/A', 'team': 'N/A', 'time': 'N/A'}

        return {
            'race_name': session.event.EventName,
            'pace_data': pace_data,
            'telemetry_data': telemetry_data,
            'winner_info': winner_info
        }, None

    except (ValueError, IndexError, KeyError) as e:
        return None, str(e)
