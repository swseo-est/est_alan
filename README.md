# EST Alan

## 개발 환경 설정

### 가상환경 설정

프로젝트를 시작하기 전에 가상환경을 설정해야 합니다. OS별로 다음 명령어를 실행하세요:

#### Windows
```bash
setup_venv.bat
```

#### macOS/Linux
```bash
./setup_venv.sh
```

## 배포
### Docker 빌드

#### Windows
```bash
build.bat
```

#### macOS/Linux
```bash
chmod +x build.sh
./build.sh
```

### 서버 log 확인

#### macOS/Linux
```bash
chmod +x logs.sh
./logs.sh -f
```

### Container 중지

#### macOS/Linux
```bash
chmod +x stop-all.sh
./stop-all.sh
```

### Container 재시작

#### macOS/Linux
```bash
chmod +x restart-all.sh
./restart-all.sh
```

