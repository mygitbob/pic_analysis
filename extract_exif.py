import os
import piexif

# Pfad zum Verzeichnis mit den Bildern
path = "/Pfad/zum/Verzeichnis"

# Schleife durch jedes Bild im Verzeichnis
for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        # Extrahieren der EXIF-Informationen
        try:
            exif_dict = piexif.load(path+"/"+file)
            print("EXIF-Informationen für " + file + ":")
            for ifd in exif_dict:
                for tag in exif_dict[ifd]:
                    print(piexif.TAGS[ifd][tag]["name"], exif_dict[ifd][tag])
        except:
            print("Keine EXIF-Informationen für " + file)

