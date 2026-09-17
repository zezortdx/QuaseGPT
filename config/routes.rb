Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check

  root "conversations#index"

  resources :conversations, only: %i[index show create destroy] do
    resources :messages, only: %i[create]
    member do
      get :stream, to: "messages#stream"
    end
  end
end
