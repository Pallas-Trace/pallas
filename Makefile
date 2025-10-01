### pipeline ###
install:
	npm install

generate:
	./node_modules/docsify-cli/bin/docsify generate docs

serve:
	./node_modules/docsify-cli/bin/docsify serve docs

clean:
	rm -rf static vendor composer.lock