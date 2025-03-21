Compiled with problems:
×
ERROR in src/components/FraudComparisonChart.tsx:88:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
    86 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
    87 |             <DatePicker
  > 88 |               label="Start Date"
       |               ^^^^^
    89 |               value={startDate}
    90 |               onChange={(newValue: Date | null) => setStartDate(newValue)}
    91 |             />
ERROR in src/components/FraudComparisonChart.tsx:97:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
     95 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
     96 |             <DatePicker
  >  97 |               label="End Date"
        |               ^^^^^
     98 |               value={endDate}
     99 |               onChange={(newValue: Date | null) => setEndDate(newValue)}
    100 |             />
ERROR in src/components/FraudTrendChart.tsx:96:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
    94 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
    95 |             <DatePicker
  > 96 |               label="Start Date"
       |               ^^^^^
    97 |               value={startDate}
    98 |               onChange={(newValue: Date | null) => setStartDate(newValue)}
    99 |             />
ERROR in src/components/FraudTrendChart.tsx:105:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
    103 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
    104 |             <DatePicker
  > 105 |               label="End Date"
        |               ^^^^^
    106 |               value={endDate}
    107 |               onChange={(newValue: Date | null) => setEndDate(newValue)}
    108 |             />
ERROR in src/components/ModelEvaluation.tsx:67:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
    65 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
    66 |             <DatePicker
  > 67 |               label="Start Date"
       |               ^^^^^
    68 |               value={startDate}
    69 |               onChange={(newValue: Date | null) => setStartDate(newValue)}
    70 |             />
ERROR in src/components/ModelEvaluation.tsx:76:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
    74 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
    75 |             <DatePicker
  > 76 |               label="End Date"
       |               ^^^^^
    77 |               value={endDate}
    78 |               onChange={(newValue: Date | null) => setEndDate(newValue)}
    79 |             />
ERROR in src/components/TransactionTable.tsx:111:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
    109 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
    110 |             <DatePicker
  > 111 |               label="Start Date"
        |               ^^^^^
    112 |               value={startDate}
    113 |               onChange={(newValue: Date | null) => setStartDate(newValue)}
    114 |             />
ERROR in src/components/TransactionTable.tsx:120:15
TS2322: Type '{ label: string; value: Date | null; onChange: (newValue: Date | null) => void; }' is not assignable to type 'IntrinsicAttributes & CustomDatePickerProps'.
  Property 'label' does not exist on type 'IntrinsicAttributes & CustomDatePickerProps'.
    118 |           <LocalizationProvider dateAdapter={AdapterDateFns}>
    119 |             <DatePicker
  > 120 |               label="End Date"
        |               ^^^^^
    121 |               value={endDate}
    122 |               onChange={(newValue: Date | null) => setEndDate(newValue)}
    123 |             />
