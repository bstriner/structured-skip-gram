from structured_skip_gram.model import SSG

def main():
    k = 10
    latent_dim = 5
    hidden_dim = 512
    SSG(x_k=k, y_k=k, latent_dim=latent_dim, hidden_dim=hidden_dim)

if __name__ == "__main__":
    main()