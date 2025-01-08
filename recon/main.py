import cv2
import os

# Diretório para salvar a imagem cadastrada
FACE_DIR = "faces"
os.makedirs(FACE_DIR, exist_ok=True)

# Carregar o modelo Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para capturar e salvar a face do usuário
def cadastrar_face():
    camera = cv2.VideoCapture(0)
    print("Posicione-se em frente à câmera. Pressione 's' para salvar.")
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Cadastro de Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Salvar apenas a região do rosto
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                cv2.imwrite(f"{FACE_DIR}/face_cadastrada.jpg", face)
                print("Face cadastrada com sucesso!")
                break
            break

    camera.release()
    cv2.destroyAllWindows()

# Função para validar a face do usuário
def validar_face():
    camera = cv2.VideoCapture(0)
    face_cadastrada = cv2.imread(f"{FACE_DIR}/face_cadastrada.jpg")

    if face_cadastrada is None:
        print("Nenhuma face cadastrada encontrada. Por favor, cadastre uma face primeiro.")
        return

    face_cadastrada_gray = cv2.cvtColor(face_cadastrada, cv2.COLOR_BGR2GRAY)
    print("Posicione-se em frente à câmera. Pressione 'q' para sair.")

    validado = False

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            # Comparação usando histogramas
            hist1 = cv2.calcHist([face_cadastrada_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([face], [0], None, [256], [0, 256])
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if similarity > 0.9 and not validado:
                print("Face validada com sucesso! Usuário autenticado.")
                validado = True

        cv2.imshow("Validação de Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Menu simples para o usuário
def menu():
    while True:
        print("\n1. Cadastrar Face")
        print("2. Validar Face")
        print("3. Sair")
        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            cadastrar_face()
        elif escolha == "2":
            validar_face()
        elif escolha == "3":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

# Executar o menu
if __name__ == "__main__":
    menu()