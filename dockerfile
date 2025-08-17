# Stage 1: Build
FROM node:20 AS build
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Production
FROM node:20 AS prod
WORKDIR /app

COPY --from=build /app/dist ./dist
COPY package*.json ./
RUN npm install --omit=dev   # only production deps

CMD ["node", "dist/index.js"]
