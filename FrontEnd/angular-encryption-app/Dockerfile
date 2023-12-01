# Etapa de desarrollo
FROM node:18-alpine AS dev-stage
WORKDIR /app
COPY package*.json ./
RUN npm install

# Etapa de construcción
FROM node:18-alpine AS builder
WORKDIR /app
COPY --from=dev-stage /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Etapa de producción
FROM nginx:alpine AS prod 
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d
EXPOSE 80
COPY --from=builder /app/dist/angular-encryption-app /usr/share/nginx/html

CMD ["nginx", "-g", "daemon off;"]
