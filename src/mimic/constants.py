gcs_eye_mapping = {
    "No Response": 1,
    "To Pain": 2,
    "To Speech": 3,
    "Spontaneously": 4,
}
gcs_verbal_mapping = {
    "No Response": 1,
    "Incomprehensible sounds": 2,
    "Inappropriate Words": 3,
    "Confused": 4,
    "Oriented": 5,
}
gcs_motor_mapping = {
    "No Response": 1,
    "Abnormal extension": 2,
    "Abnormal Flexion": 3,
    "Flex-withdraws": 4,
    "Localizes Pain": 5,
    "Obeys Commands": 6,
}
gcs_total_mapping = {
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "11": 9,
    "12": 10,
    "13": 11,
    "14": 12,
    "15": 13,
}

cap_refill_rate_mapping = {
    "0": 1,
    "1": 2,
}

stds = dict(
    [
        ("Hours", 14.396261767527724),
        ("Diastolic blood pressure", 285.80064177699705),
        ("Fraction inspired oxygen", 0.1961013042470289),
        ("Glucose", 9190.367721597377),
        ("Heart Rate", 132.41586088485442),
        ("Height", 12.332785645604897),
        ("Mean blood pressure", 266.45492092726295),
        ("Oxygen saturation", 2094.753594800329),
        ("Respiratory rate", 2025.1666030044469),
        ("Systolic blood pressure", 882.396478974552),
        ("Temperature", 12.879852903644485),
        ("Weight", 95.5778654729231),
        ("pH", 11110.745176079576),
    ]
)

means = dict(
    [
        ("Hours", 22.028152722731797),
        ("Diastolic blood pressure", 63.40139620838688),
        ("Fraction inspired oxygen", 0.5220774309673805),
        ("Glucose", 233.5193111471457),
        ("Heart Rate", 86.05173178993036),
        ("Height", 169.33463796477494),
        ("Mean blood pressure", 78.61474847093386),
        ("Oxygen saturation", 100.99360210904216),
        ("Respiratory rate", 21.34307497701275),
        ("Systolic blood pressure", 118.69927129942835),
        ("Temperature", 36.96791995122653),
        ("Weight", 84.91834694253167),
        ("pH", 130.70163154775614),
    ]
)

# first value in bin is the cutout values that we treat as missing value
bin_ranges = {
    # https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
    # <0 941,842 instances
    # 0-119 775,786 instnaces
    # 120-129 204,999 instances
    # 130-139 151,336 instances
    # 140-179 205,956 instances
    # >=180 12,336 instances
    "Systolic blood pressure": [0, 120, 130, 140, 180],
    # <0 942,208 instances
    # 0-79 1,188,103 instances
    # 80-89 95,648 instances
    # 90-119 61,476 instances
    # >=120 4,840 instances
    "Diastolic blood pressure": [0, 80, 90, 120],
    # Oxygen-enriched air has a higher FIO2 than 0.21; up to 1.00 which means 100% oxygen.
    # FIO2 is typically maintained below 0.5 even with mechanical ventilation,
    # to avoid oxygen toxicity, but there are applications when up to 100% is routinely used.
    # https://en.wikipedia.org/wiki/Fraction_of_inspired_oxygen
    # <0 2,124,492 instances
    # <0.21 824 instances
    # 0.21<0.3 880 instances
    # 0.3<0.4 17,687 instances
    # 0.4<0.5 55,093 instances
    # 0.5<0.6 49,645 instances
    # 0.6<0.7 13,784 instances
    # 0.7<0.8 8,000 instances
    # 0.8<0.9 4,652 instances
    # 0.9<1.0 2,083 instances
    # >=1.0 15,135 instances
    "Fraction inspired oxygen": [
        0,
        0.21,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    # https://www.cdc.gov/diabetes/basics/getting-tested.html
    # <0 1,984,546 instances
    # 0<90 27,612 instances
    # 90<100 23,040 instances
    # 100<125 79,210 instances
    # 125<140 43,496 instances
    # 140<200 888,566 instances
    # >=200 45,805 instances
    "Glucose": [0, 90, 100, 125, 140, 200],
    # <0 941,580
    # 0<50 14,732
    # 50<60 61,650
    # 60<70 177,829
    # 70<80 271,464
    # 80<90 303,199
    # 90<100 223,888
    # 100<110 230,482
    # 110<120 39,889
    # 120<130 16,968,
    # 130<140 6,335
    # 140<150 2,603
    # 150<160 987
    # 160<170 377
    # 170<180 149
    # 180<190 73
    # 190<200 70
    "Heart Rate": [
        0,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
    ],
    # <0 2,291,764
    # 0<150 9
    # 150<160 83
    # 160<170 148
    # 170<180 168
    # 180<190 96
    # >=190 7
    "Height": [0, 150, 160, 170, 180, 190],
    # https://en.wikipedia.org/wiki/Mean_arterial_pressure
    # <60 1,056,648
    # 60<90 968,199
    # 90<92 38,924
    # 92<96 65,063
    # >=96 163,441
    "Mean blood pressure": [60, 90, 92, 96],
    # https://www.ridgmountpractice.nhs.uk/pulse-oximeters
    # <0 931,847
    # 0<92 73,362
    # 92<93 39,927
    # 93<94 63,242
    # 94<95 94,937
    # 95<96 132,754
    # >=96 956,206
    "Oxygen saturation": [0, 92, 93, 94, 95, 96],
    # https://en.wikipedia.org/wiki/Respiratory_rate
    # <0 931,738
    # 0<12 71,123
    # 12<18 460,906
    # 18<25 593,456
    # 18<25 161,345
    # 25<30 73,707
    "Respiratory rate": [0, 12, 18, 25, 30],
    # https://en.wikipedia.org/wiki/Human_body_temperature
    # <0 1,921,856
    # 0<35 3,628
    # 35<36.5 70,077
    # 36.5<37.5 231,811
    # 37.5<38.3 50,812
    # 38.3<40 13,845
    # 40<41 200
    # >=41 46
    "Temperature": [0, 35, 36.5, 37.5, 38.3, 40, 41],
    # <0 2,233,929
    # 0<60 7,787
    # 60<70 9,545
    # 70<80 10,866
    # 80<90 9,663
    # 90<100 7,973
    # 100<110 5,244
    # 110<120 3,056
    # 120<130 1,749
    # 130<140 916
    # 140<150 599
    # >=150 948
    "Weight": [0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    # https://www.ncbi.nlm.nih.gov/books/NBK507807/#:~:text=One%20of%20these%20is%20maintaining,with%20the%20average%20at%207.40.
    # <0 2,170,783
    # 0<7.35 50,110
    # 7.35<7.4 28,549
    # 7.4<7.45 25,531
    # 7.45<7.5 12,657
    # >=7.5 4,645
    "pH": [0, 7.35, 7.4, 7.45, 7.5],
}

# row of discretized empty measurements
empty_measurements_row = [
    0,
    3,
    8,
    19,
    24,
    31,
    45,
    51,
    58,
    75,
    82,
    87,
    94,
    100,
    106,
    114,
    126,
]
