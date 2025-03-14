from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import clip
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

# Carregar variáveis de ambiente
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

def detectar_genero_imagem(image_path):
    try:
        # Forçar o uso da CPU
        device = "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Carregar a imagem
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Lista de descrições
        descriptions = [
            "uma paisagem calma", "uma cena triste", "uma festa animada",
            "uma noite chuvosa", "um pôr do sol relaxante", "uma cena assustadora",
            "uma praia ensolarada", "um ambiente futurista", "um retrato melancólico",
            "um lugar abandonado", "uma cidade movimentada", "uma floresta misteriosa",
            "um show ao vivo", "uma paisagem urbana", "um ambiente romântico"
        ]

        description_to_genre = {
            "uma paisagem calma": "lo-fi",
            "uma cena triste": "indie acústico",
            "uma festa animada": "eletrônica",
            "uma noite chuvosa": "jazz",
            "um pôr do sol relaxante": "chill-out",
            "uma cena assustadora": "dark ambient",
            "uma praia ensolarada": "reggae",
            "um ambiente futurista": "synthwave",
            "um retrato melancólico": "blues",
            "um lugar abandonado": "post-rock",
            "uma cidade movimentada": "hip-hop",
            "uma floresta misteriosa": "folk",
            "um show ao vivo": "rock",
            "uma paisagem urbana": "hip-hop",
            "um ambiente romântico": "pop",
        }

        # Converter descrições para embeddings CLIP
        text_inputs = clip.tokenize(descriptions).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

        # Calcular similaridade entre imagem e descrições
        similarities = (image_features @ text_features.T).softmax(dim=-1)
        best_match_idx = similarities.argmax().item()
        best_description = descriptions[best_match_idx]

        # Obter o gênero correspondente
        best_genre = description_to_genre.get(best_description, "desconhecido")

        # Liberar memória explicitamente
        del image, text_inputs, image_features, text_features, model, preprocess
        torch.cuda.empty_cache()  # Limpar cache da GPU (se houver)

        return best_genre

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return "desconhecido"

# Função para buscar músicas no Spotify
def buscar_musicas_spotify(genero, limite=5):
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))

        offset = random.randint(0, 100)  # Adiciona aleatoriedade
        resultados = sp.search(q=f"genre:{genero}", type="track", limit=limite, offset=offset)
        if resultados["tracks"]["items"]:
            return [(track["name"], track["artists"][0]["name"], track["external_urls"]["spotify"])
                    for track in resultados["tracks"]["items"]]
        return []

    except Exception as e:
        print(f"Erro ao buscar músicas no Spotify: {e}")
        return []

# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Salvar o arquivo
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Detectar o gênero da imagem
        genero = detectar_genero_imagem(filepath)

        # Buscar músicas no Spotify
        limite = int(request.form.get('limite', 5))  # Número de músicas
        musicas = buscar_musicas_spotify(genero, limite)

        return render_template('index.html', genero=genero, musicas=musicas, image_url=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
