using Pkg                                  # διαχείριση πακέτων
Pkg.activate(@__DIR__)                      # ενεργοποίηση του περιβάλλοντος του webapp
Pkg.add(["Oxygen", "HTTP", "JSON3", "Images", "FileIO", "ImageIO", "ImageMagick", "ImageMorphology", "StaticArrays"])   # εξαρτήσεις του Live Service
# το τοπικό Agents (με SocialForce και D* Lite) δεν μεταγλωττίζεται με νεότερο Distributions (@check_args)· καρφιτσώνεται η έκδοση του evac3
Pkg.add(PackageSpec(name = "Distributions", version = "0.25.123"))
Pkg.pin("Distributions")
Pkg.develop(path = joinpath(@__DIR__, "..", "Agents"))   # το τοπικό Agents που χρησιμοποιούν τα σενάρια
Pkg.instantiate()                           # εγκατάσταση όλων των εξαρτήσεων
Pkg.precompile()                            # προ-μεταγλώττιση ώστε η πρώτη εκκίνηση να είναι γρήγορη
