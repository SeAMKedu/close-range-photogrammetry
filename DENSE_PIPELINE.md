# Tiheä laserstereo – 29.10.2026

Uusi haara tuottaa kalibroiduista Basler-kuvapareista tiheät syvyyskartat,
tarkistaa havaintojen luotettavuuden, tarkentaa kolmen asennon kohdistuksen
ja muodostaa yhden yhteisen TSDF-pintaverkon kustakin threepiece-esineestä.

## Blender-tiedostot

Tulokset ovat kansiossa `2026-10-29/dense_results/blender`:

- `dense_threepiece.blend`: viisi threepiece-esinettä, yksi yhteinen pinta kustakin.
- `dense_single.blend`: yhden asennon kokeet; Piiri_single sisältää kaksi erillistä kuvaushetkeä.
- `Sini_ennen_jalkeen.blend`: harva ja tiheä threepiece-pinta rinnakkain samalla mittakaavalla.
- `Sini_mesh_vertailu.blend`: saman tiheän single-pistepilven TSDF-, Ball Pivoting- ja rajattu Poisson-pinta.
- Esinekohtaiset tiedostot, esimerkiksi `Sini_threepiece.blend`.
- PNG-esikatselut, mukaan lukien viisto näkymä yhdistetyistä pinnoista.

Pistepilvi on jokaisessa esinekohtaisessa tiedostossa aluksi piilotettu vain
**silmäkuvakkeella**. Näytä se Outlinerista ja piilota pinta toisesta silmästä.
Kokoelmaa ei tarvitse erikseen ottaa käyttöön näyttökuvakkeesta.
Pisteet säilyvät oikeina mesh-vertex-koordinaatteina; Geometry Nodes näyttää
niissä 0,10 mm säteiset merkit. Merkkien koko ei kuvaa mittausepävarmuutta.

Blenderin mittakaava on millimetri. Kaikkiin tuloksiin tehdään sama aiemmin
arvioitu 9,8° oikaisu, eikä esineiden geometriaa litistetä tasoon.

## Tulokset

21 Basler-kuvaparia (42 kuvaa), yhteensä **2 991 835 hyväksyttyä syvyyshavaintoa**.
Jokaisen kuvan pikselihavainto ei ole riippumaton mittaus: stereomenetelmä käyttää
naapurustoja ja säännöllistystä. Seuraavat pisteet ovat 0,2 mm voxel-suodatuksen
jälkeisiä yhdistettyjä havaintoja.

| Threepiece-esine | Aiemmat harvat pisteet | Tiheät yhdistetyt pisteet | TSDF-kolmioita |
| --- | ---: | ---: | ---: |
| Harmu | 1 167 | 62 518 | 60 005 |
| Metal | 617 | 174 771 | 156 776 |
| Musti | 69 | 52 248 | 50 154 |
| Piiri | 971 | 72 378 | 67 753 |
| Sini | 468 | 203 583 | 191 169 |

Hyväksyttyjen syvyyspikselien osuus vasemman kuvan kappalemaskista:

- Harmu: noin 85 %.
- Metal: noin 94–97 %.
- Musti: noin 44–61 %.
- Piiri: noin 46–52 %.
- Sini: noin 52–66 %.

Osuudet ovat kuva-alan kattavuutta suhteessa automaattiseen maskiin; ne eivät
ole koko kappaleen pinnan kattavuuksia. Maski voi sisältää esimerkiksi reiän
sisäpuolen, jossa kelvollista syvyyttä ei kuulu saada. Takapintoja ei ole havaittu.

## Menetelmä

1. **Kalibroitu oikaisu.** Käytössä ovat juuren `calibration_files`-kansion
   20.10.2026 kalibroinnit Basler 96:lle ja 07:lle. Käyttäjä vahvisti asetelman
   ja tarkennusten säilyneen. Oikaisu käyttää `stereoRectify(flags=0, alpha=1)`:
   kuvien pääpisteet saavat erota, mikä on tässä sisäänpäin suunnattujen kameroiden
   asetelmassa hyödyllistä. Dispariteetit ovat negatiivisia; niitä ei hylätä
   pelkän etumerkin perusteella. Q-matriisi käsittelee pääpiste-eron.
2. **Kalibroinnin tarkistus nykyisistä kuvista.** Molempiin suuntiin tarkistetut
   SIFT-vastinparit antavat oikaistujen kuvien pystysuuntaisen virheen. Sini_single:
   mediaani 0,115 px ja 95. persentiili 0,575 px, 58 uskottavaa vastinparia.
   Mustin kuvissa piirteitä on liian vähän vahvaan erilliseen tarkistukseen;
   nämä tulokset on merkitty tarkistettaviksi. Tämä ei ole uusi kalibrointi.
3. **Tiheä stereo.** OpenCV StereoSGBM, 7 px ikkuna, uniqueness ratio 10.
   Dispariteettihaun alue arvioidaan harvan rekonstruktion kappalehavainnoista.
   Jos Basler-pari antaa liian vähän havaintoja syvyysrajan arviointiin,
   aiempaa Hikrobot-parin kalibroitua harvaa aineistoa käytetään lisätukena.
   Varsinainen tiheä vastinpistehaku tehdään vain Basler-kuvista.
4. **Luotettavuussuodatus.** Kappalemaski molemmissa kuvissa, vasen–oikea-virhe
   enintään 1 px, paikallinen normalisoitu korrelaatio vähintään 0,50,
   riittävä paikallinen kontrasti, syvyysrajat ja pienten irrallisten pikselialueiden
   poisto. Epävarmoja pikseleitä ei täytetä pelkällä kuvan tasoituksella.
5. **Kolmen asennon kohdistus.** Aiempi esineen piirrepohjainen tai
   ääriviivapohjainen muunnos alustaa robustin piste–taso-ICP:n. Korjaus hyväksytään
   vain rajatun liikkeen ja molempiin suuntiin arvioidun peiton perusteella.
   Tasainen pinta yksin ei ratkaise kaikkia liikesuuntia, joten alustuksen
   merkitys säilyy. Mustin toisen asennon harva syvyyssiirtymä osoittautui vääräksi;
   se alustettiin uudelleen ääriviivan ja tiheän syvyystiedon perusteella.
6. **Yhteinen pinta.** Kunkin asennon suodatettu syvyyskartta integroidaan
   samaan Open3D TSDF-tilavuuteen. Voxel on 0,25 mm, katkaisuetäisyys 1,0 mm.
   Kamera-asennot ja millimetriyksiköt huomioidaan integraatiossa. Lopputuloksena
   on yksi yhteinen mesh kustakin threepiece-sarjasta.

## Mesh-vertailu ja rajat

Sini_single-kokeessa tuotettiin TSDF, Ball Pivoting ja Poisson. Poissonista
poistettiin heikon tiheyden kärjet ja yli 0,6 mm lähimmästä hyväksytystä
havainnosta olevat kärjet. Poisson täyttää enemmän alueita, mutta se myös
interpoloidaan havaintojen väliin. Sileys tai pienempi reikämäärä ei osoita
parempaa mittatarkkuutta. TSDF on päävientien oletus, koska se käyttää
syvyyskarttoja ja kameranäkymiä suoraan.

Pinnat ovat aiempaa kattavampia ja yksityiskohtaisempia, mutta niissä on edelleen
kohinaa, reikiä ja peittyviä alueita. Musti vaatii erityistä visuaalista tarkistusta.
Kohdistuksen ja pintatuen virheluvut ovat sisäisiä tarkistuksia, eivät riippumatonta
mittatarkkuuden todentamista. Uusi haarakin käyttää nykyistä kalibrointia ja
olemassa olevien harvojen rekonstruktioiden alustuksia.

Seuraava aineistoparannus olisi useampia kappaleen kiertoasentoja sekä pysyviä
kohdistusmerkkejä tai erillisiä ilman projektoria otettuja kohdistuskuvia.
Paikallaan olevan projektorin pisteet vaihtavat paikkaa kappaleen pinnalla,
joten niitä ei voi olettaa pysyviksi eri asentojen välillä.

## Uusinta-ajo

Projektin juuresta:

```powershell
.venv/Scripts/python.exe -u run_dense_laser_pipeline.py
```

Komento tarkistaa välimuistin asetukset, kalibrointimuunnokset ja lähdekuvien
SHA256-tunnisteet, yhdistää aineistot, tekee mesh-vertailun, vie Blender-tiedostot
ja suorittaa tarkistukset. Tulostiedostot korvautuvat.

Laske myös stereo uudelleen:

```powershell
.venv/Scripts/python.exe -u run_dense_laser_pipeline.py --recompute
```

Blender-polun voi antaa parametrilla `--blender`. Oletus on tämän koneen Blender 5.0.
Toteutus on konfiguroitu tämän projektin 29.10.2026 aineistolle, ja se edellyttää
aiempia `laser_results/run_report.json`, `fusion_report.json`, `leveling_report.json`
sekä kuvaushetkikohtaisia harvoja pistepilviä.

## Tarkistukset ja raportit

- `dense_report.json`: 21 kuvan syvyystulokset, maskikattavuus ja kalibrointidiagnostiikka.
- `models_report.json`: 11 mallia, kohdistukset ja pintojen tuki havainnoissa.
- `*/frame_*/rectification_check.jpg`: oikaistut kamerakuvat kuvariviviivoineen.
- `*/frame_*/coverage_overlay.jpg`: hylätyt maskin pikselit punaisina.
- `*/frame_*/dense_frame.npz`: syvyys, validiteettimaski, dispariteetti ja 3D-pisteet.
- `models/*/model_report.json`: käytetyt asennot ja kohdistuksen laatuluvut.
- `verification.json`: koordinaattimuunnosten, synteettisen tasokokeen ja tiedostojen tarkistukset.
- `blender/verification.json`: tallennettujen tiedostojen avaus ja pistepilvien silmäkytkimet.
- `pipeline_execution.json` ja `logs`: uusinta-ajon vaiheiden paluukoodit ja lokit.

Synteettisessä tarkistuksessa tunnettu 3D-geometria palautuu projektion ja
Q-muunnoksen läpi alle 1e-8 mm virheellä. Kahden keinotekoisen syvyyskameran
TSDF-tasokokeen 95 % virhe on alle 0,3 mm. Nämä testaavat ohjelman
koordinaatistoja ja yksiköitä, eivät todellisten kameroiden mittatarkkuutta.
