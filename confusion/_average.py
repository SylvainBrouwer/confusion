#micro averages the results for a list of confusion matrices
def micro_average(confs, beta=1):
    epsilon = 1e-8
    result = [0 ,0 ,0 ,0]
    for conf in confs:
        result[0] += conf.TP
        result[1] += conf.FP
        result[2] += conf.TN
        result[3] += conf.FN
    prec = result[0]/(result[0]+result[1]+epsilon)
    rec = result[0]/(result[0]+result[3]+epsilon)
    acc = (result[0]+result[2])/(result[0]+result[1]+result[2]+result[3]+epsilon)
    f = (1+beta*beta)*((prec*rec)/(beta*beta*prec+rec+epsilon))
    return (prec, rec, f, acc)

#macro averages the results for a list of confusion matrices
def macro_average(confs, beta):
    result = [0 ,0, 0, 0]
    for conf in confs:
        result[0]  += conf.precision()
        result[1]  += conf.recall()
        result[2]  += conf.f_beta(beta)
        result[3]  += conf.accuracy()
    return tuple(x/len(confs) for x in result)