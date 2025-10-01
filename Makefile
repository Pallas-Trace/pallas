### pipeline ###
install:
	npm install

generate:
	./node_modules/docsify-cli/bin/docsify generate

serve:
	./node_modules/docsify-cli/bin/docsify serve

clean:
	rm -rf static vendor composer.lock