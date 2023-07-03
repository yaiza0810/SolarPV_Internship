import GreENERLib as gr
hey = gr.physicalform()
dt = gr.data()
# dt.create_df()
pred = hey.calcule_pred()
real = dt.df_prod_toit
eval = gr.eval()
# eval.plot_result(real, pred)
# eval.evaluation(real, pred)
# print('\nOndulateur A \n')
# real_A = dt.df_inverters['Onduleur A: energie totale (MWh)']
# power = [[8.1+13.77, 139], [8.1+14.58, -41]]


# pred_A = hey.inverters(power_angles=power, model_irrad='isotropic')
# eval.plot_result(real_A, pred_A, 'figs/onduleurA.png')
# eval.evaluation(real_A, pred_A)

# print('\nOndulateur DA \n')
# real_DA = dt.df_inverters['Onduleur D-A: energie totale (MWh)']
# powerDA =[32.8, 139]
# pred_DA = hey.inverters(power_angles=powerDA, model_irrad='isotropic')
# eval.plot_result(real_DA, pred_DA, 'figs/onduleurDA.png')
# eval.evaluation(real_DA, pred_DA)


pso = gr.PSO()
df= pso.df
pred, true = pso.predictions()
eval.plot_result(true, pred, 'figs/pso.png')
eval.evaluation(true, pred)



# lstm = gr.Model()
# df = dt.weather_data
# real_lstm, pred_lstm = lstm.prediction(df)
# eval.evaluation(real_lstm, pred_lstm)


