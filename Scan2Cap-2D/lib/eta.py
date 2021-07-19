def decode_eta(eta_sec):
    eta = {'h': 0, 'm': 0, 's': 0}
    if eta_sec < 60:
        eta['s'] = int(eta_sec)
    elif 60 <= eta_sec < 3600:
        eta['m'] = int(eta_sec / 60)
        eta['s'] = int(eta_sec % 60)
    else:
        eta['h'] = int(eta_sec / (60 * 60))
        eta['m'] = int(eta_sec % (60 * 60) / 60)
        eta['s'] = int(eta_sec % (60 * 60) % 60)

    return eta
