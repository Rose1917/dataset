clean:
	rm -rf ae_100k_*.json
tar:
	pip3 freeze >requirements.txt
	tar -czf code.tar.gz *.py utils/
