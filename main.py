from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.DEBUG)
app.logger.addHandler(handler)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        app.logger.debug('Received data: %s', data)

        # Process input data
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['wu_consumption']

        # Prepare the Prophet model
        m = Prophet()
        app.logger.debug('Prophet model initialized')

        # Add regressors if available
        if 'visitor_count' in df.columns:
            df['visitor_count'] = df['visitor_count'].fillna(0)
            m.add_regressor('visitor_count')
            app.logger.debug('Added visitor_count regressor')
        if 'user_count' in df.columns:
            df['user_count'] = df['user_count'].fillna(0)
            m.add_regressor('user_count')
            app.logger.debug('Added user_count regressor')

        # Fit the model
        m.fit(df[['ds', 'y', 'visitor_count', 'user_count']].dropna(subset=['y']))
        app.logger.debug('Model fitting complete')

        # Create a DataFrame for future predictions
        future = m.make_future_dataframe(periods=5, freq='M')

        # Include regressor values for the future (if available)
        if 'visitor_count' in df.columns and 'user_count' in df.columns:
            avg_visitor_count = df['visitor_count'].mean()
            avg_user_count = df['user_count'].mean()
            future['visitor_count'] = avg_visitor_count
            future['user_count'] = avg_user_count
            app.logger.debug('Added future values for regressors')

        # Generate the forecast
        forecast = m.predict(future)
        app.logger.debug('Forecast generation complete')

        # Extract and return the relevant part of the forecast
        forecasted_values = forecast[['ds', 'yhat']].tail(5)
        result = forecasted_values.to_dict(orient='records')
        app.logger.debug('Forecast data prepared for response')
        return jsonify(result)
    except Exception as e:
        app.logger.error('Error during forecasting: %s', e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
