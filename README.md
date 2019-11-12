
# Multi Strike Quotation Matrix API


## Quotation Matrix
POST /quotation_matrix/<int:tenor> 

    wget http://127.0.0.1:5000/quotation_matrix/90  --post-data=string

## Term Sheet
POST /term_sheet 

    wget http://127.0.0.1:5000/term_sheet  --post-data='{"tenor":90,"ticker":"SCB","notional":500000,"discount":0.97}' --header='Content-Type:application/json'
