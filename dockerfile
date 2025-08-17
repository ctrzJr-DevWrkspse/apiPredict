FROM node:18-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Serve production build
FROM node:18-alpine AS prod
WORKDIR /app

COPY --from=build /app/dist ./dist
COPY package*.json ./

RUN npm install vite --omit=dev

EXPOSE 3000
CMD ["npm", "run", "start"]
