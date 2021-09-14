''' predict.py

'''

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from modules.preprocess import genome
from modules.extract import features, dimentional_reduction
from modules.utils import longest_orf, load_param, load_model

# suppress warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def from_genome(filename, params, model, outfile, n_genomes=None, seq='all', OFFSET=30, noise=False, LMIN=60):
    # preprocess genome
    _cds, _ncs = genome(filename=filename,n_genomes=n_genomes,
                        seq=seq, OFFSET=OFFSET,noise=noise, LMIN=LMIN)
    
    # seqrecord2fasta
    tmp_file = "./input/tmp.fna"
    _seq = _cds + _ncs
    seqrecord2fasta(tmp_file, _seq)
    
    # from_fasta
    predictions = from_fasta(tmp_file, params, model, outfile)
    
    return predictions


def from_fasta(filename, params, model, outfile):   
    # extract sequences from fasta file
    records = fasta2seqrecords(filename)
    
    # extract features
    feat, _ = features(records)
    
    # dim red features
    w, p = load_param(params)
    nn_input = dimentional_reduction(feat, w, p)
    
    # predict
    predictions = predict(nn_input, model)
    
    save_predictions(outfile, predictions, records)
    
    return predictions


def seqrecord2fasta(filename, seq_records):
    fasta_records = []
    
    for i,record in enumerate(seq_records):
        r = SeqRecord(record.seq, description=f"Seq No {i+1}", name="", id="")
        fasta_records.append(r)

    SeqIO.write(fasta_records, filename, "fasta")
    print(f"Saved genome fragments to {filename}.")


def fasta2seqrecords(filename):
    seq_records = []
    
    try:
        for frag in SeqIO.parse(filename, "fasta"):
            r = SeqRecord(frag.seq, description=frag.description)
            seq_records.append(r)
    except Exception:
        print(f"Error: no file named {filename}.")
        return 0
        
    return seq_records
    
    
def predict(data, model):
    model = load_model(model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    prediction = model.predict(data)
    return prediction


def save_predictions(outfile, predictions, seq_records):  
    results = []
    
    for (pred, record) in zip(predictions, seq_records):
        if pred > 0.5:
            orf = longest_orf(record.seq)
            SEQ = record[orf[0]:orf[1]].seq
            r = SeqRecord(SEQ, id=record.id, name=record.name, description=record.description)
            results.append(r)
            
    SeqIO.write(results, outfile, "fasta")
    print(f"Saved genome fragments to {outfile}.")