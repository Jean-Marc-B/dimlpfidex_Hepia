


    elif model == "MLP_Patch":
        patch_input = Input(shape=(height, width, nbChannels))

        x = Flatten()(patch_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        coord_input = Input(shape=(2,))
        y = Dense(8, activation='relu')(coord_input)

        merged = Concatenate()([x, y])
        merged = Dense(64, activation='relu')(merged)
        output = Dense(nb_classes, activation='softmax')(merged)

        model = Model(inputs=[patch_input, coord_input], outputs=output)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()


# Si on fait le nouveau réseau qui mix VGG et métadatas : (tester son accuracy avant...)

# 23 données

# Créer des fichiers de metadonnées pour train, test, etc.
# Il nous faut toujours les 2 données (train, stats)
# Problème : les patchs se créés à partir de l'image... donc les métadonnées ne sont pas présents...
# Les métadonnées peuvent aider à apprendre et sont liées à chaque patch/à chaque image selon si on entraine avec les patchs ou non.
# Il faut juste appeler train_cnn avec un couple (image, metadatas)

# Si on voudra train avec les patchs il faudra encore modifier MLP_Patch genre pour ajouter les métadatas...
# On utilise MLP_Patch pour gagner du temps car c'est extrêmement long à entrainer...


#TODO : Trouver comment récupérer les métadonnées et les ajouter aux données
