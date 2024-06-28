
# branche SZ3-stable

## Contient l'adaptation de l'API de SZ2 à SZ3

- Le Makefile force le paquet SZ3 à l'installation, à changer si besoin

- les variables qui peuvent aider à la compilation de pallas :
```bash
cmake .. \
        -DBUILD_DOC=FALSE \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR/pallas" \
        -DCMAKE_BUILD_TYPE="$PALLAS_COMPILE_MODE" \
        -DSZ3_ROOT_DIR="$INSTALL_DIR/SZ3" \
        -DSZ3_INCLUDE_DIRS="$INSTALL_DIR/SZ3/include" \
        -DSZ3_LIBRARIES="$INSTALL_DIR/SZ3/lib/libSZ3c.so" \
        -DZFP_ROOT_DIR="$INSTALL_DIR/ZFP" \
        -DZSTD_INCLUDE_DIRS="$INSTALL_DIR/zstd/include" \
        -DZSTD_LIBRARIES="$INSTALL_DIR/zstd/lib/libzstd.so"
        2>&1 | tee "$ORIGINAL_DIR"/pallas.config
```

- N'a actuellement que les commits du dépot principal jusqu'au 24/06 car une modification dans la semaine semble générer un bug après le merge.

- Le taux de compression est réglable par la variable *conf.absErrorBound* dans pallas_storage.cpp


# branche buffer-dev

## Contient les modification de SZ3-stable + bufferisation I/O sur les timestamps pour compresser par plus gros blocs.

- compress_read / compress_write sont maintenant des methode de la classe **File**

- **BufferFile** est un classe dérivée de **File** ayant un buffer pour gérer les timestamps

- compress_read / compress_write ont les même entrée et sortie qu'auparavant mais avec un buffer qui limite les I/O.

- La taille du buffer est réglable par le define *SIZE_BUFFER_TIMESTAMP* dans pallas_storage.cpp

- Le format des fichiers contenant les durations est modifié si un algorithme de compression est actif (uncompressedSize -> compressedSize -> compressedArray)

- Un bug est encore présent dans cette version si aucun algorithme de compression n'est actif, je n'ai pas eu le temps de debuger ceci.

- L'encodage n'est pas supporté.

- La documention des fonctions modifié n'ont pas toujours été mise à jour.