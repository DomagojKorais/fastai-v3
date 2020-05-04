import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import wikipedia

export_file_url = 'https://www.googleapis.com/drive/v3/files/1KhHDCrX0HtIA1Cx7bNJs7utotmu6sg5y?alt=media&key=AIzaSyC5XjgwohRcXHOr9tv84KCb-DQugVdUPaU'
export_file_name = 'italian_birds_resnet34_fine_tuned'

classes = ['accipiter_nisus', 'actitis_hypoleucos', 'aegithalos_caudatus', 'alcedo_atthis', 'anas_clypeata', 'anas_crecca', 'anas_platyrhynchos', 'aquila_chrysaetos', 'ardea_alba', 'ardea_cinerea', 'ardea_purpurea', 'ardeola_ralloides', 'arenaria_interpres', 'asio_flammeus', 'asio_otus', 'athene_noctua', 'aythya_ferina', 'botaurus_stellaris', 'bubulcus_ibis', 'buteo_buteo', 'calidris_alpina', 'carduelis_carduelis', 'carduelis_chloris', 'carduelis_spinus', 'charadrius_dubius', 'charadrius_morinellus', 'ciconia_ciconia', 'cinclus_cinclus', 'circus_aeruginosus', 'circus_pygargus', 'cisticola_juncidis', 'coccothraustes_coccothraustes', 'coracias_garrulus', 'cygnus_olor', 'dendrocopos_major', 'egretta_garzetta', 'emberiza_schoeniclus', 'erithacus_rubecula', 'falco_naumanni', 'falco_peregrinus', 'falco_tinnunculus', 'falco_vespertinus', 'fringilla_coelebs', 'fringilla_montifringilla', 'fulica_atra', 'gallinago_gallinago', 'gallinula_chloropus', 'garrulus_glandarius', 'gypaetus_barbatus', 'gyps_fulvus', 'haliaeetus_albicilla', 'himantopus_himantopus', 'hirundo_rustica', 'ixobrychus_minutus', 'jynx_torquilla', 'lagopus_mutus', 'lanius_collurio', 'loxia_curvirostra', 'merops_apiaster', 'milvus_migrans', 'milvus_milvus', 'montifringilla_nivalis', 'motacilla_alba', 'motacilla_cinerea', 'muscicapa_striata', 'nycticorax_nycticorax', 'oenanthe_oenanthe', 'pandion_haliaetus', 'parus_ater', 'parus_caeruleus', 'parus_cristatus', 'parus_major', 'parus_palustris', 'phalacrocorax_carbo', 'philomachus_pugnax', 'phoenicopterus_roseus', 'phoenicurus_ochruros', 'phoenicurus_phoenicurus', 'phylloscopus_collybita', 'picus_viridis', 'platalea_leucorodia', 'plegadis_falcinellus', 'podiceps_cristatus', 'porzana_parva', 'prunella_modularis', 'rallus_aquaticus', 'recurvirostra_avosetta', 'regulus_regulus', 'remiz_pendulinus', 'saxicola_torquata', 'sitta_europaea', 'strix_aluco', 'sylvia_atricapilla', 'sylvia_melanocephala', 'tachybaptus_ruficollis', 'tetrao_tetrix', 'threskiornis_aethiopica', 'tringa_glareola', 'tringa_nebularia', 'troglodytes_troglodytes', 'turdus_merula', 'upupa_epops', 'vanellus_vanellus']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = str(learn.predict(img)[0]).replace("_", " ").title()
    page = wikipedia.page(prediction)
    return JSONResponse({'result': prediction,'url':page.url })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
