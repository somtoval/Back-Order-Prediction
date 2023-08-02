from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            national_inv = float(request.form.get('national_inv')),
            lead_time = float(request.form.get('lead_time')),
            in_transit_qty = float(request.form.get('in_transit_qty')),
            forecast_3_month = float(request.form.get('forecast_3_month')),
            forecast_6_month = float(request.form.get('forecast_6_month')),
            forecast_9_month = float(request.form.get('forecast_9_month')),
            sales_1_month = float(request.form.get('sales_1_month')),
            sales_3_month = float(request.form.get('sales_3_month')),
            sales_6_month = float(request.form.get('sales_6_month')),
            sales_9_month = float(request.form.get('sales_9_month')),
            min_bank = float(request.form.get('min_bank')),
            potential_issue = float(request.form.get('potential_issue')),
            pieces_past_due = float(request.form.get('pieces_past_due')),
            perf_6_month_avg = float(request.form.get('perf_6_month_avg')),
            perf_12_month_avg = float(request.form.get('perf_12_month_avg')),
            local_bo_qty = float(request.form.get('local_bo_qty')),
            deck_risk = float(request.form.get('deck_risk')),
            oe_constraint = float(request.form.get('oe_constraint')),
            ppap_risk = float(request.form.get('ppap_risk')),
            stop_auto_buy = float(request.form.get('stop_auto_buy')),
            rev_stop = float(request.form.get('rev_stop')),
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = int(predict_pipeline.predict(final_new_data))
        print('This the the prediction: >>>>', pred)
        if pred == 1:
            pred = 'Yes, Back order will occur'
        elif pred == 0:
            pred = 'No, Back order not will occur'

        return render_template('index.html', output = pred)


if __name__=='__main__':
    app.run(host='0.0.0.0')