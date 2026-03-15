FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY backend/ backend/
COPY public/ public/

EXPOSE 3000

CMD ["node", "backend/main.js"]
