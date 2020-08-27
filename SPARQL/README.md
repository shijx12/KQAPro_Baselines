## Requirements
- python3
- rdflib=4.2.2
> We need to fix some bugs of rdflib.
> In rdflib/rdflib/plugins/sparql/parser.py, Line 68, we should modify this line to
```if i + 1 < l and (not isinstance(terms[i + 1], str) or terms[i + 1] not in ".,;"):```
> In rdflib/rdflib/plugins/serializers/turtle.py, Line 328, we should change `use_plain=True` to `use_plain=False`

- SPARQLWrapper=1.8.4
> Note: must not install keepalive, which may use the available ports out
- virtuoso backend to run sparql query

## How to install virtuoso backend
We take Ubuntu system as an example. 

1.download and install virtuoso into our system.
```
git clone https://github.com/openlink/virtuoso-opensource.git Virtuoso-Opensource
cd Virtuoso-Opensource
git checkout stable/7
sudo apt-get install libtool gawk gperf autoconf automake libtool flex bison m4 make openssl libssl-dev
sudo ./autogen.sh
sudo ./configure
sudo make
sudo make install
```

2.modify some necessary configs:
```
sudo useradd virtuoso --home /usr/local/virtuoso-opensource
sudo chown -R virtuoso /usr/local/virtuoso-opensource
cd /usr/local/virtuoso-opensource/var/lib/virtuoso/db
sudo vim virtuoso.ini
```
change `CheckpointInterval` from default 60 to 0, to avoid automatical checkpoint process which will cause 404 error.

and start up the virtuoso service:
```
sudo -H -u virtuoso ../../../../bin/virtuoso-t -f &
```
Now you can access the service via the default port 8890.
Enter `[ip]:8890` in a browser, you will see the virtuoso page.

3.prepare our generated graph `kg.ttl`:
```
sudo chmod 777 kg.ttl
sudo mv kg.ttl /usr/local/virtuoso-opensource/share/virtuoso/vad
```

4.enter terminal
```
cd /usr/local/virtuoso-opensource/bin
sudo ./isql
```

5.execute following commands in terminal:
```
SPARQL CREATE GRAPH <[graph name]>;
SPARQL CLEAR GRAPH <[graph name]>;
delete from db.dba.load_list;
ld_dir('/usr/local/virtuoso-opensource/share/virtuoso/vad', 'kg.ttl', '[graph name]');
rdf_loader_run();
select * from DB.DBA.load_list;
exit;
```
`[graph name]` should be replace with your graph name.
You are success if `rdf_loader_run()` lasts for about 10 seconds.


## How to run
1. run 
```python3 sparql_engine.py --kb_path <dataset/kb.json> --ttl_path <results/kg.ttl>```
and get `<results/kg.ttl>`
2. load `<results/kg.ttl>` into the virtuoso backend
3. modify `virtuoso_address` and `virtuoso_graph_uri` in `sparql_engine.py` to make them point to your virtuoso service
4. preprocess the training data
```
python -m SPARQL.preprocess --input_dir ./dataset --output_dir /data/sjx/exp/KBQA/SPARQL
cp ./dataset/kb.json /data/sjx/exp/KBQA/SPARQL
```
5. sparql engine has been prepared well, now you can run `train.py` with following command
```
CUDA_VISIBLE_DEVICES=0 python -m SPARQL.train --input_dir /data/sjx/exp/KBQA/SPARQL --save_dir /data/sjx/exp/KBQA/SPARQL/debug
```
6. predict answer for test set
```
CUDA_VISIBLE_DEVICES=2 python -m SPARQL.predict --input_dir /data/sjx/exp/KBQA/SPARQL --save_dir /data/sjx/exp/KBQA/SPARQL/debug
```
