def calculate_congestion(num, dis, tm):
    congestion = ""
    cnt = num
    con = (cnt / 54) * 100

    if dis <= 24:
        if con <= 50:
            congestion = "여유"
        elif con <= 90:
            congestion = "보통"
        elif con <= 100:
            congestion = "혼잡"
        else:
            congestion = "매우 혼잡"
    else:
        if con <= 50:
            congestion = "여유"
        elif con <= 85:
            congestion = "보통"
        elif con <= 95:
            congestion = "혼잡"
        else:
            congestion = "매우 혼잡"

    # return {'weather': tm,
    #         'congestion': congestion,
    #         'peopleNumber': num}
    # print(dis, tm, congestion, num)
    return congestion

# for i in range(20, 50):
#     calculate_congestion(i, , 18)
