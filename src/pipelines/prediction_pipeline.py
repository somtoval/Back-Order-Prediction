import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            logging.info(f'Entered Prediction')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info(f'Loading Preprocessor')
            preprocessor = load_object(preprocessor_path)
            logging.info(f'Loading model')
            model = load_object(model_path)

            logging.info(f'Entered Transformation')
            data_scaled = preprocessor.transform(features)
            logging.info(f'Finished Transformation and entering prediction')
            pred = model.predict(data_scaled)
            logging.info(f'Finished Prediction')
            
            return pred
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 national_inv,
                 lead_time,
                 in_transit_qty,
                 forecast_3_month,
                 forecast_6_month,
                 forecast_9_month,
                 sales_1_month,
                 sales_3_month,
                 sales_6_month,
                 sales_9_month,
                 min_bank,
                 potential_issue,
                 pieces_past_due,
                 perf_6_month_avg,
                 perf_12_month_avg,
                 local_bo_qty,
                 deck_risk,
                 oe_constraint,
                 ppap_risk,
                 stop_auto_buy,
                 rev_stop
                 ) -> None:
        self.national_inv = national_inv
        self.lead_time = lead_time
        self.in_transit_qty = in_transit_qty
        self.forecast_3_month = forecast_3_month
        self.forecast_6_month = forecast_3_month
        self.forecast_9_month = forecast_9_month
        self.sales_1_month = sales_1_month
        self.sales_3_month = sales_3_month
        self.sales_6_month = sales_6_month
        self.sales_9_month = sales_9_month
        self.min_bank = min_bank
        self.potential_issue = potential_issue
        self.pieces_past_due = pieces_past_due
        self.perf_6_month_avg = perf_6_month_avg
        self.perf_12_month_avg = perf_12_month_avg
        self.local_bo_qty = local_bo_qty
        self.deck_risk = deck_risk
        self.oe_constraint = oe_constraint
        self.ppap_risk = ppap_risk
        self.stop_auto_buy = stop_auto_buy
        self.rev_stop = rev_stop

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'national_inv':[self.national_inv],
                'lead_time':[self.lead_time],
                'in_transit_qty':[self.in_transit_qty],
                'forecast_3_month':[self.forecast_3_month],
                'forecast_6_month':[self.forecast_6_month],
                'forecast_9_month':[self.forecast_9_month],
                'sales_1_month':[self.sales_1_month],
                'sales_3_month':[self.sales_3_month],
                'sales_6_month':[self.sales_6_month],
                'sales_9_month':[self.sales_9_month],
                'min_bank':[self.min_bank],
                'potential_issue':[self.potential_issue],
                'pieces_past_due':[self.pieces_past_due],
                'perf_6_month_avg':[self.perf_6_month_avg],
                'perf_12_month_avg':[self.perf_12_month_avg],
                'local_bo_qty':[self.local_bo_qty],
                'deck_risk':[self.deck_risk],
                'oe_constraint':[self.oe_constraint],
                'ppap_risk':[self.ppap_risk],
                'stop_auto_buy':[self.stop_auto_buy],
                'rev_stop':[self.rev_stop]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)