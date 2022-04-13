from scipy.io import loadmat
#  from scipy.io import savemat
#  import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

cond = loadmat(r"D:\MATLAB\a_experiment\Number\GenCfg\cfgs.mat")  # h5py.File
number = np.squeeze(cond['cfgs']['number'][0, 0])
avgrad = np.squeeze(cond['cfgs']['radavg'][0, 0])
category = (avgrad-20)/10
nclass = len(set(category))
kernel = np.cos(np.linspace(-np.pi, np.pi, nclass*2-1))

number_pool = np.append(np.array(1), np.arange(2, 15, 2))
nnumber = len(number_pool)
nrepeat = 2
nfold = 5
lda = LinearDiscriminantAnalysis()

data = loadmat(r"D:\MATLAB\a_data\size_number\cf_clear.mat")
data = data['eeg_clear']

marker = data[0, 0]['trialinfo']
trial_idx = np.squeeze(marker == 8)
eeg = np.squeeze(data[0, 0]['trial'])[trial_idx]
eeg_tar = np.stack(eeg, 2)

label = data[0, 0]['label']
label = label.tolist()
chan = [['P7'], ['P5'], ['P3'], ['P1'], ['Pz'], ['P2'],
        ['P4'], ['P6'], ['P8'], ['PO7'], ['PO3'], ['POz'],
        ['PO4'], ['PO8'], ['O1'], ['Oz'], ['O2']]
elect_idx = [label.index(chan[i]) for i in range(len(chan))]
eeg_tar_occ = eeg_tar[elect_idx, :, :]
ntime = np.shape(eeg_tar_occ)[1]
ntrial = np.shape(eeg_tar_occ)[2]

evidence = np.full([nnumber, ntime], np.nan)
for i in range(nnumber):
    number_idx = number == number_pool[i]
    tmpeeg = eeg_tar_occ[:, :, number_idx]  # 17*600*200
    tmpclass = category[number_idx]

    nobsvn = len(tmpclass)
    obsvn_pool = np.arange(nobsvn)
    nperfold = (np.ceil(nobsvn/nfold)).astype(np.int8)

    for j in range(ntime):
        ieeg = tmpeeg[:, j, :]  # 17*200
        page = np.full([nclass, nclass, nfold, nrepeat], np.nan)
        for k in range(nrepeat):
            rest = obsvn_pool
            for m in range(nfold):
                perfold_idx = []
                if m == max(range(nfold)):
                    perfold_idx = rest
                else:
                    perfold_idx = np.random.choice(rest, nperfold, replace=False)
                restfold_idx = np.delete(obsvn_pool, perfold_idx)
                rest = np.setdiff1d(rest, perfold_idx)

                test_data = ieeg[:, perfold_idx].T
                tran_data = ieeg[:, restfold_idx].T
                test_mark = tmpclass[perfold_idx]
                tran_mark = tmpclass[restfold_idx]

                predproba = lda.fit(tran_data, tran_mark).predict_proba(test_data)
                unit = np.full([nclass, nclass], np.nan)
                for n in range(nclass):
                    probaByClass = predproba[test_mark == (n+1), :]
                    if np.any(probaByClass == np.nan):
                        print('wrong1')
                    unit[:, n] = np.nanmean(probaByClass, 0)  # 5*5
                page[:, :, m, k] = unit

        if np.any(page == np.nan):
            print('wrong1')
        knit = np.nanmean(np.nanmean(page, 2), 2)
        band = np.full([nclass], np.nan)
        for k in range(nclass):
            band[k] = np.dot(knit[k], kernel[nclass-1-k:nclass*2-1-k])
        evidence[i, j] = np.nanmean(band)

plt.plot(evidence[1, :])
plt.plot(evidence[4, :])
plt.plot(evidence[7, :])
plt.show()