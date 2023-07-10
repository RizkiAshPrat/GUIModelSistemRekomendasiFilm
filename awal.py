from flask import Flask, render_template, request, send_file
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import requests
from io import BytesIO

app = Flask(__name__)

#Mengambil Data User dari github dan dibuat Dataframe
penggunaFold=np.load('Model/IPWR(K=10, C=25, F=10)/Pengguna_fold2.npy', allow_pickle=True)
loadFilm=pd.read_csv('Model/u.item', sep='|', index_col=0, header=None, encoding='iso-8859-1')


def RekomendasiFilmPCC(TargetPengguna, TargetTopN):
    MemberCluster=np.load('Model/PCC(K=20, C=25, F=10)/AHC_rank20_25_fold2.npy', allow_pickle=True)
    #Mengubah supaya ukuran setiap Membernya sama, mengisi nan
    max_row_length = max(len(row) for row in MemberCluster)
    MemberCluster_fixed = np.array([row + [np.nan] * (max_row_length - len(row)) for row in MemberCluster])

    #Get Cluster of User, 2 dimensi Cluster dan Index dalam cluster
    indicesCluster = np.argwhere(MemberCluster_fixed == TargetPengguna)
    
    #Cek Irisan per cluster, apakah pengguna target menjadi bagian cluster (Pengguna mana yang diprediksi didalam cluster)
    Irisan=np.intersect1d(penggunaFold, MemberCluster[indicesCluster[0][0]])
    JumlahTargetPengguna=len(Irisan)
    indicesIndexInCluster = np.argwhere(Irisan == TargetPengguna)

    #Rekomendasi
    rekomendasiFilm=np.load('Model/PCC(K=20, C=25, F=10)/Top-100_Item_PCC_fold2.npy', allow_pickle=True)

    hasilRekom=rekomendasiFilm[indicesCluster[0][0]][indicesIndexInCluster[0][0]][:TargetTopN]
    hasilRekomJudul=[]
    for TopN in range(len(hasilRekom)):
        if np.isnan(hasilRekom[TopN]):
            hasilRekomJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilRekom[TopN]].values
            judul=dataFilm[0]
            hasilRekomJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilRekomJudulEnum = list(enumerate(hasilRekomJudul, start=1))
    

    #Item Training
    itemTraining=np.load('Model/PCC(K=20, C=25, F=10)/ItemTrainingPCC_fold2.npy', allow_pickle=True)
    hasilItemTraining=itemTraining[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTrainingJudul=[]
    for TopN in range(len(hasilItemTraining)):
        if np.isnan(hasilItemTraining[TopN]):
            hasilItemTrainingJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTraining[TopN]].values
            judul=dataFilm[0]
            hasilItemTrainingJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTrainingJudulEnum=list(enumerate(hasilItemTrainingJudul, start=1))

    #Item Test
    itemTest=np.load('Model/PCC(K=20, C=25, F=10)/ItemTestPCC_fold2.npy', allow_pickle=True)
    hasilItemTest=itemTest[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTestJudul=[]
    for TopN in range(len(hasilItemTest)):
        if np.isnan(hasilItemTest[TopN]):
            hasilItemTestJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTest[TopN]].values
            judul=dataFilm[0]
            hasilItemTestJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTestJudulEnum=list(enumerate(hasilItemTestJudul, start=1))

    #Mencari irisan item test dan hasil rekomendasi
    irisanItemTestdanrekom=list(set(hasilItemTest).intersection(set(hasilRekom)))
    
    #Mengurutkan irisan berdasarkan indexnya di hasilRekom
    sorted_irisanItemTestdanrekom = sorted(irisanItemTestdanrekom, key=lambda x: hasilRekom.index(x))
    #Get Judul Irisan
    hasilIrisanItemTestdanRekomJudul=[]
    for TopN in range(len(sorted_irisanItemTestdanrekom)):
        dataFilm=loadFilm.loc[sorted_irisanItemTestdanrekom[TopN]].values
        judul=dataFilm[0]
        hasilIrisanItemTestdanRekomJudul.append(judul)
    
    indicesIrisaninRekom=[]
    #Perulangan untuk mencari index setiap irisan antara hasil rekomendasi dengan item Test
    for indicesIrisan in (sorted_irisanItemTestdanrekom):
        indicesIrisaninRekom.extend(np.argwhere(np.array(hasilRekom) == indicesIrisan).flatten().tolist())

    #MetEval sesuai Target Pengguna, Karena disimpan urut per cluster, sehingga perlu dicari dahulu. Dia ada di cluster mana, kemudian jumlah target pengguna cluster sebelumnya berapa lalu dia ada di urutan keberapa pada cluster itu
    #Jika berada di Cluster 1 maka tinggal ambil dia di cluster 1 urutan keberapa
    Clusters=indicesCluster[0][0]
    if Clusters == 0:
        indicesTarget=indicesIndexInCluster[0][0]
    else:
        totalTargetSebelum=0
        #Perulangan sebanyak cluster sebelumnya
        for i in range(Clusters):
            MemberClustersTarget=len(rekomendasiFilm[i])
            totalTargetSebelum+=MemberClustersTarget
        #Jika sudah diketahui jumlah target pengguna di cluster sebelumnya maka dijumlah dia diposisi berapa di cluster itu
        indicesTarget=totalTargetSebelum+indicesIndexInCluster[0][0]
    LoadNDCGPCC=np.load('Model/PCC(K=20, C=25, F=10)/NDCGPCC_fold2.npy')
    NDCGPCC=LoadNDCGPCC[indicesTarget]

    return hasilRekomJudulEnum, len(hasilRekomJudulEnum), hasilItemTrainingJudulEnum, len(hasilItemTrainingJudulEnum), hasilItemTestJudulEnum, len(hasilItemTestJudulEnum), hasilIrisanItemTestdanRekomJudul, len(hasilIrisanItemTestdanRekomJudul), NDCGPCC[TargetTopN-1]

def RekomendasiFilmAdjCos(TargetPengguna, TargetTopN):
    MemberCluster=np.load('Model/AdjCos(K=10, C=30, F=30)/AHC_rank10_30_fold2.npy', allow_pickle=True)
    #Mengubah supaya ukuran setiap Membernya sama, mengisi nan
    max_row_length = max(len(row) for row in MemberCluster)
    MemberCluster_fixed = np.array([row + [np.nan] * (max_row_length - len(row)) for row in MemberCluster])

    #Get Cluster of User, 2 dimensi Cluster dan Index dalam cluster
    indicesCluster = np.argwhere(MemberCluster_fixed == TargetPengguna)
    
    #Cek Irisan per cluster, apakah pengguna target menjadi bagian cluster (Pengguna mana yang diprediksi didalam cluster)
    Irisan=np.intersect1d(penggunaFold, MemberCluster[indicesCluster[0][0]])
    JumlahTargetPengguna=len(Irisan)
    indicesIndexInCluster = np.argwhere(Irisan == TargetPengguna)

    #Rekomendasi
    rekomendasiFilm=np.load('Model/AdjCos(K=10, C=30, F=30)/Top-100_Item_AdjCos_fold2.npy', allow_pickle=True)

    hasilRekom=rekomendasiFilm[indicesCluster[0][0]][indicesIndexInCluster[0][0]][:TargetTopN]
    hasilRekomJudul=[]
    for TopN in range(len(hasilRekom)):
        if np.isnan(hasilRekom[TopN]):
            hasilRekomJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilRekom[TopN]].values
            judul=dataFilm[0]
            hasilRekomJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilRekomJudulEnum = list(enumerate(hasilRekomJudul, start=1))
    

    #Item Training
    itemTraining=np.load('Model/AdjCos(K=10, C=30, F=30)/ItemTrainingAdjCos_fold2.npy', allow_pickle=True)
    hasilItemTraining=itemTraining[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTrainingJudul=[]
    for TopN in range(len(hasilItemTraining)):
        if np.isnan(hasilItemTraining[TopN]):
            hasilItemTrainingJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTraining[TopN]].values
            judul=dataFilm[0]
            hasilItemTrainingJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTrainingJudulEnum=list(enumerate(hasilItemTrainingJudul, start=1))

    #Item Test
    itemTest=np.load('Model/AdjCos(K=10, C=30, F=30)/ItemTestAdjCos_fold2.npy', allow_pickle=True)
    hasilItemTest=itemTest[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTestJudul=[]
    for TopN in range(len(hasilItemTest)):
        if np.isnan(hasilItemTest[TopN]):
            hasilItemTestJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTest[TopN]].values
            judul=dataFilm[0]
            hasilItemTestJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTestJudulEnum=list(enumerate(hasilItemTestJudul, start=1))

    #Mencari irisan item test dan hasil rekomendasi
    irisanItemTestdanrekom=list(set(hasilItemTest).intersection(set(hasilRekom)))
    
    #Mengurutkan irisan berdasarkan indexnya di hasilRekom
    sorted_irisanItemTestdanrekom = sorted(irisanItemTestdanrekom, key=lambda x: hasilRekom.index(x))
    #Get Judul Irisan
    hasilIrisanItemTestdanRekomJudul=[]
    for TopN in range(len(sorted_irisanItemTestdanrekom)):
        dataFilm=loadFilm.loc[sorted_irisanItemTestdanrekom[TopN]].values
        judul=dataFilm[0]
        hasilIrisanItemTestdanRekomJudul.append(judul)
    
    indicesIrisaninRekom=[]
    #Perulangan untuk mencari index setiap irisan antara hasil rekomendasi dengan item Test
    for indicesIrisan in (sorted_irisanItemTestdanrekom):
        indicesIrisaninRekom.extend(np.argwhere(np.array(hasilRekom) == indicesIrisan).flatten().tolist())

    #MetEval sesuai Target Pengguna, Karena disimpan urut per cluster, sehingga perlu dicari dahulu. Dia ada di cluster mana, kemudian jumlah target pengguna cluster sebelumnya berapa lalu dia ada di urutan keberapa pada cluster itu
    #Jika berada di Cluster 1 maka tinggal ambil dia di cluster 1 urutan keberapa
    Clusters=indicesCluster[0][0]
    if Clusters == 0:
        indicesTarget=indicesIndexInCluster[0][0]
    else:
        totalTargetSebelum=0
        #Perulangan sebanyak cluster sebelumnya
        for i in range(Clusters):
            MemberClustersTarget=len(rekomendasiFilm[i])
            totalTargetSebelum+=MemberClustersTarget
        #Jika sudah diketahui jumlah target pengguna di cluster sebelumnya maka dijumlah dia diposisi berapa di cluster itu
        indicesTarget=totalTargetSebelum+indicesIndexInCluster[0][0]
    #LoadNDCGPCC=np.load(BytesIO(requests.get('https://github.com/RizkiAshPrat/ImplementasiSkripsi/raw/main/NDCGAdjCos_fold2.npy').content))
    LoadNDCGPCC=np.load('Model/AdjCos(K=10, C=30, F=30)/NDCGAdjCos_fold2.npy')
    NDCGPCC=LoadNDCGPCC[indicesTarget]

    return hasilRekomJudulEnum, len(hasilRekomJudulEnum), hasilItemTrainingJudulEnum, len(hasilItemTrainingJudulEnum), hasilItemTestJudulEnum, len(hasilItemTestJudulEnum), hasilIrisanItemTestdanRekomJudul, len(hasilIrisanItemTestdanRekomJudul), NDCGPCC[TargetTopN-1]

def RekomendasiFilmITR(TargetPengguna, TargetTopN):
    MemberCluster=np.load('Model/ITR(K=10, C=30, F=10)/AHC_rank10_30_fold2.npy', allow_pickle=True)
    #Mengubah supaya ukuran setiap Membernya sama, mengisi nan
    max_row_length = max(len(row) for row in MemberCluster)
    MemberCluster_fixed = np.array([row + [np.nan] * (max_row_length - len(row)) for row in MemberCluster])

    #Get Cluster of User, 2 dimensi Cluster dan Index dalam cluster
    indicesCluster = np.argwhere(MemberCluster_fixed == TargetPengguna)
    
    #Cek Irisan per cluster, apakah pengguna target menjadi bagian cluster (Pengguna mana yang diprediksi didalam cluster)
    Irisan=np.intersect1d(penggunaFold, MemberCluster[indicesCluster[0][0]])
    JumlahTargetPengguna=len(Irisan)
    indicesIndexInCluster = np.argwhere(Irisan == TargetPengguna)

    #Rekomendasi
    rekomendasiFilm=np.load('Model/ITR(K=10, C=30, F=10)/Top-100_Item_ITR_fold2.npy', allow_pickle=True)

    hasilRekom=rekomendasiFilm[indicesCluster[0][0]][indicesIndexInCluster[0][0]][:TargetTopN]
    hasilRekomJudul=[]
    for TopN in range(len(hasilRekom)):
        if np.isnan(hasilRekom[TopN]):
            hasilRekomJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilRekom[TopN]].values
            judul=dataFilm[0]
            hasilRekomJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilRekomJudulEnum = list(enumerate(hasilRekomJudul, start=1))
    

    #Item Training
    itemTraining=np.load('Model/ITR(K=10, C=30, F=10)/ItemTrainingITR_fold2.npy', allow_pickle=True)
    hasilItemTraining=itemTraining[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTrainingJudul=[]
    for TopN in range(len(hasilItemTraining)):
        if np.isnan(hasilItemTraining[TopN]):
            hasilItemTrainingJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTraining[TopN]].values
            judul=dataFilm[0]
            hasilItemTrainingJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTrainingJudulEnum=list(enumerate(hasilItemTrainingJudul, start=1))

    #Item Test
    itemTest=np.load('Model/ITR(K=10, C=30, F=10)/ItemTestITR_fold2.npy', allow_pickle=True)
    hasilItemTest=itemTest[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTestJudul=[]
    for TopN in range(len(hasilItemTest)):
        if np.isnan(hasilItemTest[TopN]):
            hasilItemTestJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTest[TopN]].values
            judul=dataFilm[0]
            hasilItemTestJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTestJudulEnum=list(enumerate(hasilItemTestJudul, start=1))

    #Mencari irisan item test dan hasil rekomendasi
    irisanItemTestdanrekom=list(set(hasilItemTest).intersection(set(hasilRekom)))
    
    #Mengurutkan irisan berdasarkan indexnya di hasilRekom
    sorted_irisanItemTestdanrekom = sorted(irisanItemTestdanrekom, key=lambda x: hasilRekom.index(x))
    #Get Judul Irisan
    hasilIrisanItemTestdanRekomJudul=[]
    for TopN in range(len(sorted_irisanItemTestdanrekom)):
        dataFilm=loadFilm.loc[sorted_irisanItemTestdanrekom[TopN]].values
        judul=dataFilm[0]
        hasilIrisanItemTestdanRekomJudul.append(judul)
    
    indicesIrisaninRekom=[]
    #Perulangan untuk mencari index setiap irisan antara hasil rekomendasi dengan item Test
    for indicesIrisan in (sorted_irisanItemTestdanrekom):
        indicesIrisaninRekom.extend(np.argwhere(np.array(hasilRekom) == indicesIrisan).flatten().tolist())

    #MetEval sesuai Target Pengguna, Karena disimpan urut per cluster, sehingga perlu dicari dahulu. Dia ada di cluster mana, kemudian jumlah target pengguna cluster sebelumnya berapa lalu dia ada di urutan keberapa pada cluster itu
    #Jika berada di Cluster 1 maka tinggal ambil dia di cluster 1 urutan keberapa
    Clusters=indicesCluster[0][0]
    if Clusters == 0:
        indicesTarget=indicesIndexInCluster[0][0]
    else:
        totalTargetSebelum=0
        #Perulangan sebanyak cluster sebelumnya
        for i in range(Clusters):
            MemberClustersTarget=len(rekomendasiFilm[i])
            totalTargetSebelum+=MemberClustersTarget
        #Jika sudah diketahui jumlah target pengguna di cluster sebelumnya maka dijumlah dia diposisi berapa di cluster itu
        indicesTarget=totalTargetSebelum+indicesIndexInCluster[0][0]
    LoadNDCGPCC=np.load('Model/ITR(K=10, C=30, F=10)/NDCGITR_fold2.npy')
    NDCGPCC=LoadNDCGPCC[indicesTarget]

    return hasilRekomJudulEnum, len(hasilRekomJudulEnum), hasilItemTrainingJudulEnum, len(hasilItemTrainingJudulEnum), hasilItemTestJudulEnum, len(hasilItemTestJudulEnum), hasilIrisanItemTestdanRekomJudul, len(hasilIrisanItemTestdanRekomJudul), NDCGPCC[TargetTopN-1]

def RekomendasiFilmIPWR(TargetPengguna, TargetTopN):
    MemberCluster=np.load('Model/IPWR(K=10, C=25, F=10)/AHC_rank10_25_fold2.npy', allow_pickle=True)
    #Mengubah supaya ukuran setiap Membernya sama, mengisi nan
    max_row_length = max(len(row) for row in MemberCluster)
    MemberCluster_fixed = np.array([row + [np.nan] * (max_row_length - len(row)) for row in MemberCluster])

    #Get Cluster of User, 2 dimensi Cluster dan Index dalam cluster
    indicesCluster = np.argwhere(MemberCluster_fixed == TargetPengguna)
    
    #Cek Irisan per cluster, apakah pengguna target menjadi bagian cluster (Pengguna mana yang diprediksi didalam cluster)
    Irisan=np.intersect1d(penggunaFold, MemberCluster[indicesCluster[0][0]])
    JumlahTargetPengguna=len(Irisan)
    indicesIndexInCluster = np.argwhere(Irisan == TargetPengguna)

    #Rekomendasi
    rekomendasiFilm=np.load('Model/IPWR(K=10, C=25, F=10)/Top-100_Item_IPWR_fold2.npy', allow_pickle=True)

    hasilRekom=rekomendasiFilm[indicesCluster[0][0]][indicesIndexInCluster[0][0]][:TargetTopN]
    hasilRekomJudul=[]
    for TopN in range(len(hasilRekom)):
        if np.isnan(hasilRekom[TopN]):
            hasilRekomJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilRekom[TopN]].values
            judul=dataFilm[0]
            hasilRekomJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilRekomJudulEnum = list(enumerate(hasilRekomJudul, start=1))
    

    #Item Training
    itemTraining=np.load('Model/IPWR(K=10, C=25, F=10)/ItemTrainingIPWR_fold2.npy', allow_pickle=True)
    hasilItemTraining=itemTraining[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTrainingJudul=[]
    for TopN in range(len(hasilItemTraining)):
        if np.isnan(hasilItemTraining[TopN]):
            hasilItemTrainingJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTraining[TopN]].values
            judul=dataFilm[0]
            hasilItemTrainingJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTrainingJudulEnum=list(enumerate(hasilItemTrainingJudul, start=1))

    #Item Test
    itemTest=np.load('Model/IPWR(K=10, C=25, F=10)/ItemTestIPWR_fold2.npy', allow_pickle=True)
    hasilItemTest=itemTest[indicesCluster[0][0]][indicesIndexInCluster[0][0]]
    hasilItemTestJudul=[]
    for TopN in range(len(hasilItemTest)):
        if np.isnan(hasilItemTest[TopN]):
            hasilItemTestJudul.append('nan')
        else:
            dataFilm=loadFilm.loc[hasilItemTest[TopN]].values
            judul=dataFilm[0]
            hasilItemTestJudul.append(judul)
    #Enumerate untuk mendapatkan nilai index, indexnya diberikan mulai 1, tapi nilai tetep dari 0
    hasilItemTestJudulEnum=list(enumerate(hasilItemTestJudul, start=1))

    #Mencari irisan item test dan hasil rekomendasi
    irisanItemTestdanrekom=list(set(hasilItemTest).intersection(set(hasilRekom)))
    
    #Mengurutkan irisan berdasarkan indexnya di hasilRekom
    sorted_irisanItemTestdanrekom = sorted(irisanItemTestdanrekom, key=lambda x: hasilRekom.index(x))
    #Get Judul Irisan
    hasilIrisanItemTestdanRekomJudul=[]
    for TopN in range(len(sorted_irisanItemTestdanrekom)):
        dataFilm=loadFilm.loc[sorted_irisanItemTestdanrekom[TopN]].values
        judul=dataFilm[0]
        hasilIrisanItemTestdanRekomJudul.append(judul)
    
    indicesIrisaninRekom=[]
    #Perulangan untuk mencari index setiap irisan antara hasil rekomendasi dengan item Test
    for indicesIrisan in (sorted_irisanItemTestdanrekom):
        indicesIrisaninRekom.extend(np.argwhere(np.array(hasilRekom) == indicesIrisan).flatten().tolist())

    #MetEval sesuai Target Pengguna, Karena disimpan urut per cluster, sehingga perlu dicari dahulu. Dia ada di cluster mana, kemudian jumlah target pengguna cluster sebelumnya berapa lalu dia ada di urutan keberapa pada cluster itu
    #Jika berada di Cluster 1 maka tinggal ambil dia di cluster 1 urutan keberapa
    Clusters=indicesCluster[0][0]
    if Clusters == 0:
        indicesTarget=indicesIndexInCluster[0][0]
    else:
        totalTargetSebelum=0
        #Perulangan sebanyak cluster sebelumnya
        for i in range(Clusters):
            MemberClustersTarget=len(rekomendasiFilm[i])
            totalTargetSebelum+=MemberClustersTarget
        #Jika sudah diketahui jumlah target pengguna di cluster sebelumnya maka dijumlah dia diposisi berapa di cluster itu
        indicesTarget=totalTargetSebelum+indicesIndexInCluster[0][0]
    LoadNDCGPCC=np.load('Model/IPWR(K=10, C=25, F=10)/NDCGIPWR_fold2.npy')
    NDCGPCC=LoadNDCGPCC[indicesTarget]

    return hasilRekomJudulEnum, len(hasilRekomJudulEnum), hasilItemTrainingJudulEnum, len(hasilItemTrainingJudulEnum), hasilItemTestJudulEnum, len(hasilItemTestJudulEnum), hasilIrisanItemTestdanRekomJudul, len(hasilIrisanItemTestdanRekomJudul), NDCGPCC[TargetTopN-1]


@app.route('/', methods=['GET','POST'])
def hasilRekomendasi():
    if request.method == 'POST':
        # Ambil data dari formulir jika ada
        inputPengguna = int(request.form.get('pengguna'))
        inputTopN = int(request.form.get('TopN'))

        #PCC
        ambilDataPCC=RekomendasiFilmPCC(inputPengguna, inputTopN)
        ambilhasilRekomJudulPCC=ambilDataPCC[0]
        ##Jika Tidak ada rekom, hasil nan
        if ambilhasilRekomJudulPCC[0][1]=='nan':
            ambilhasilRekomJudulPCC=ambilhasilRekomJudulPCC[:1]
        ambiljumlahHasilRekomJudulPCC=ambilDataPCC[1]
        ambilhasilItemTrainingJudulPCC=ambilDataPCC[2]
        ambiljumlahHasilItemTrainingJudulPCC=ambilDataPCC[3]
        ambilhasilItemTestJudulPCC=ambilDataPCC[4]
        ambiljumlahHasilItemTestJudulPCC=ambilDataPCC[5]
        ambilhasilIrisanItemTestdanRekomJudulPCC=ambilDataPCC[6]
        ambiljumlahhasilIrisanItemTestdanRekomJudulPCC=ambilDataPCC[7]
        ambilNDCGPCC=np.round(ambilDataPCC[8], 5)

        #AdjCos
        ambilDataAdjCos=RekomendasiFilmAdjCos(inputPengguna, inputTopN)
        ambilhasilRekomJudulAdjCos=ambilDataAdjCos[0]
        ##Jika Tidak ada rekom, hasil nan
        if ambilhasilRekomJudulAdjCos[0][1]=='nan':
            ambilhasilRekomJudulAdjCos=ambilhasilRekomJudulAdjCos[:1]
        ambiljumlahHasilRekomJudulAdjCos=ambilDataAdjCos[1]
        ambilhasilItemTrainingJudulAdjCos=ambilDataAdjCos[2]
        ambiljumlahHasilItemTrainingJudulAdjCos=ambilDataAdjCos[3]
        ambilhasilItemTestJudulAdjCos=ambilDataAdjCos[4]
        ambiljumlahHasilItemTestJudulAdjCos=ambilDataAdjCos[5]
        ambilhasilIrisanItemTestdanRekomJudulAdjCos=ambilDataAdjCos[6]
        ambiljumlahhasilIrisanItemTestdanRekomJudulAdjCos=ambilDataAdjCos[7]
        ambilNDCGAdjCos=np.round(ambilDataAdjCos[8], 5)

        #ITR
        ambilDataITR=RekomendasiFilmITR(inputPengguna, inputTopN)
        ambilhasilRekomJudulITR=ambilDataITR[0]
        ambiljumlahHasilRekomJudulITR=ambilDataITR[1]
        ambilhasilItemTrainingJudulITR=ambilDataITR[2]
        ambiljumlahHasilItemTrainingJudulITR=ambilDataITR[3]
        ambilhasilItemTestJudulITR=ambilDataITR[4]
        ambiljumlahHasilItemTestJudulITR=ambilDataITR[5]
        ambilhasilIrisanItemTestdanRekomJudulITR=ambilDataITR[6]
        ambiljumlahhasilIrisanItemTestdanRekomJudulITR=ambilDataITR[7]
        ambilNDCGITR=np.round(ambilDataITR[8], 5)

        #IPWR
        ambilDataIPWR=RekomendasiFilmIPWR(inputPengguna, inputTopN)
        ambilhasilRekomJudulIPWR=ambilDataIPWR[0]
        ambiljumlahHasilRekomJudulIPWR=ambilDataIPWR[1]
        ambilhasilItemTrainingJudulIPWR=ambilDataIPWR[2]
        ambiljumlahHasilItemTrainingJudulIPWR=ambilDataIPWR[3]
        ambilhasilItemTestJudulIPWR=ambilDataIPWR[4]
        ambiljumlahHasilItemTestJudulIPWR=ambilDataIPWR[5]
        ambilhasilIrisanItemTestdanRekomJudulIPWR=ambilDataIPWR[6]
        ambiljumlahhasilIrisanItemTestdanRekomJudulIPWR=ambilDataIPWR[7]
        ambilNDCGIPWR=np.round(ambilDataIPWR[8], 5)

        # Lakukan sesuatu dengan data yang diterima
        # Misalnya, tampilkan data di konsol
        print(inputPengguna)
        print(inputTopN)
        # Render template form.html dengan hasil data
        return render_template("/hasilRekomendasi.html", pengguna=penggunaFold, inputPengguna=inputPengguna, inputTopN=inputTopN, hasilRekomJudulPCC=ambilhasilRekomJudulPCC, ambiljumlahHasilRekomJudulPCC=ambiljumlahHasilRekomJudulPCC, hasilItemTrainingJudulPCC=ambilhasilItemTrainingJudulPCC, ambiljumlahHasilItemTrainingJudulPCC=ambiljumlahHasilItemTrainingJudulPCC, hasilItemTestJudulPCC=ambilhasilItemTestJudulPCC, ambiljumlahHasilItemTestJudulPCC=ambiljumlahHasilItemTestJudulPCC, ambilhasilIrisanItemTestdanRekomJudulPCC=ambilhasilIrisanItemTestdanRekomJudulPCC, ambiljumlahhasilIrisanItemTestdanRekomJudulPCC=ambiljumlahhasilIrisanItemTestdanRekomJudulPCC, hasilNDCGPCC=ambilNDCGPCC, 
                               hasilRekomJudulAdjCos=ambilhasilRekomJudulAdjCos, ambiljumlahHasilRekomJudulAdjCos=ambiljumlahHasilRekomJudulAdjCos, hasilItemTrainingJudulAdjCos=ambilhasilItemTrainingJudulAdjCos, ambiljumlahHasilItemTrainingJudulAdjCos=ambiljumlahHasilItemTrainingJudulAdjCos, hasilItemTestJudulAdjCos=ambilhasilItemTestJudulAdjCos, ambiljumlahHasilItemTestJudulAdjCos=ambiljumlahHasilItemTestJudulAdjCos, ambilhasilIrisanItemTestdanRekomJudulAdjCos=ambilhasilIrisanItemTestdanRekomJudulAdjCos, ambiljumlahhasilIrisanItemTestdanRekomJudulAdjCos=ambiljumlahhasilIrisanItemTestdanRekomJudulAdjCos, hasilNDCGAdjCos=ambilNDCGAdjCos,
                               hasilRekomJudulITR=ambilhasilRekomJudulITR, ambiljumlahHasilRekomJudulITR=ambiljumlahHasilRekomJudulITR, hasilItemTrainingJudulITR=ambilhasilItemTrainingJudulITR, ambiljumlahHasilItemTrainingJudulITR=ambiljumlahHasilItemTrainingJudulITR, hasilItemTestJudulITR=ambilhasilItemTestJudulITR, ambiljumlahHasilItemTestJudulITR=ambiljumlahHasilItemTestJudulITR, ambilhasilIrisanItemTestdanRekomJudulITR=ambilhasilIrisanItemTestdanRekomJudulITR, ambiljumlahhasilIrisanItemTestdanRekomJudulITR=ambiljumlahhasilIrisanItemTestdanRekomJudulITR, hasilNDCGITR=ambilNDCGITR,
                               hasilRekomJudulIPWR=ambilhasilRekomJudulIPWR, ambiljumlahHasilRekomJudulIPWR=ambiljumlahHasilRekomJudulIPWR, hasilItemTrainingJudulIPWR=ambilhasilItemTrainingJudulIPWR, ambiljumlahHasilItemTrainingJudulIPWR=ambiljumlahHasilItemTrainingJudulIPWR, hasilItemTestJudulIPWR=ambilhasilItemTestJudulIPWR, ambiljumlahHasilItemTestJudulIPWR=ambiljumlahHasilItemTestJudulIPWR, ambilhasilIrisanItemTestdanRekomJudulIPWR=ambilhasilIrisanItemTestdanRekomJudulIPWR, ambiljumlahhasilIrisanItemTestdanRekomJudulIPWR=ambiljumlahhasilIrisanItemTestdanRekomJudulIPWR, hasilNDCGIPWR=ambilNDCGIPWR)
    
    # Tampilkan halaman formulir jika metode adalah GET atau jika terdapat data hasil
    return render_template("/hasilRekomendasi.html", pengguna=penggunaFold)

# home
@app.route("/")
def home():
    return render_template("hasilRekomendasi.html", pengguna=penggunaFold)

# metode
@app.route("/#metode")
def metode():
    return render_template("#metode")

# about
@app.route("/about.html")
def about():
    return render_template("about.html")
   
if __name__== "__main__":
    app.run()