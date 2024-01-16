#!/usr/bin/env python3

import freesasa
import Bio
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from Bio.PDB import NeighborSearch
from collections import defaultdict
from collections import Counter
import pymol
import pandas as pd
import numpy as np
import sys
import io
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import subprocess
import argparse
from joblib import load


class PDB_setup:
    def __init__(self, wildtype, mutation, position, **kwargs):
        # check for proper input
        if position is None:
            raise ValueError("Position must be supplied during Initialization.")
        if mutation is None:
            raise ValueError("Mutation must be supplied during Initialization.")
        if wildtype is None:
            raise Warning("Wildtype is not supplied so we infer Wildtype from supplied position.")

        if wildtype.upper() == mutation.upper():
            raise ValueError("No valid mutation provided. Same Wildtype as mutation.")

        self.pos = position
        self.wt = wildtype.upper() if wildtype is not None else self._infer_wildtype()
        self.mut = mutation.upper()

        for key, value in kwargs.items():
            setattr(self, key, value)
            print(f"{key} = {value}")

        if len(self.wt) == 1 or len(self.mut) == 1:
            # we try to convert it to standard input.
            try:
                self.uniformize_input()
            except:
                raise ValueError(f"Input: {self.wt} or {self.mut} could not be converted")

        self.dhydrophob = None
        self.weight_change = None
        self.size_change = None
        self.pdb_file_wt_path = None
        self.domain_info = None
        self.domain_num = None
        self.domain_offset = None
        self.SAP = None
        self.conservation = None
        self.propensity = None
        self.ddg = None
        self.cost_dict = None
        self.mut_df = "./Supplementary/full_dataset_ready.csv"
        self.combined_scores = None
        self.rot = None
        self.rot_ex = None
        self.rot_super_stric = None
        self.trgt_2 = None
        self.mut_pdb_save = None
        self.cost = None
        self.clash_wt = None
        self.clash_mut = None
        self.clashcounter_diff = None
        self.dhydro_dsize_dweigth = None
        self.input = None
        self.probability = None
        self.feature_importance = None
        self.AMIVA_F = None
        self.explainer = None
        self.shap_values = None
        self.analysis_input = None
        self.probability = None
        self.expl_importance = None
        self.predicted_class = None

    def uniformize_input(self):

        # convert user input into valid downstream usable values.

        wildtype = self.wt
        mutation = self.mut

        reverse_trans_aa_dict = {"ALA": "A", "GLY": "G", "SER": "S", "THR": "T", "CYS": "C", "MET": "M", "VAL": "V",
                                 "ILE": "I",
                                 "LEU": "L", "PHE": "F", "TYR": "Y", "TRP": "W", "HIS": "H", "GLN": "Q", "GLU": "E",
                                 "ASN": "N",
                                 "ASP": "D", "PRO": "P", "LYS": "K", "ARG": "R"}

        reverse_1_letter_aa_trans_dict = {v: k for k, v in reverse_trans_aa_dict.items()}

        accepted_aas = list(reverse_1_letter_aa_trans_dict.values())

        if len(wildtype) == 1:
            self.wt = reverse_1_letter_aa_trans_dict[wildtype]
        if len(mutation) == 1:
            self.mut = reverse_1_letter_aa_trans_dict[mutation]

        if self.mut and self.wt in accepted_aas:
            print(f"Input was converted to standard {self.wt}, {self.mut}")
        else:
            exit(1)

    def probability_pred(self, training_feature_importance=False):

        AMIVA_F = self.AMIVA_F
        self.input = self.prediction_input_prep()

        proba_predictions = AMIVA_F.predict_proba(self.input)

        feature_importances = AMIVA_F.feature_importances_

        if training_feature_importance:
            print(feature_importances)

        max_probability_index = np.argmax(proba_predictions)
        max_probability_value = proba_predictions[0][max_probability_index]

        result = None
        if max_probability_index == 1:
            result = "pathological"
        else:
            result = "benign"

        self.probability = round(max_probability_value, 2)

        self.expl_importance = round(max_probability_value, 2) * feature_importances

        self.predicted_class = result
        # {'pathological': 1, 'benign': 0}

        print(f"AMIVA_F predicts: {result} with associated probability: {round(max_probability_value, 2)}")

        if round(max_probability_value, 2) < 0.55:
            print(f"Warning! Prediction was very close and confidence is small.")

        # feature_contribution_dict = dict(zip(X.columns, individual_feature_contributions))
        # sorted_contributions = sorted(feature_contribution_dict.items(), key=lambda x: x[1], reverse=True)

        # return proba_predictions, sorted_contributions

    def _compute_clashcountdiff(self):

        wt_clashes = 0 #self.clash_wt[1] #this should always be 0.
        mut_clashes = self.clash_mut[1]

        print(f"{wt_clashes=}")
        print(f"{mut_clashes=}")
        print(f"{np.abs(wt_clashes - mut_clashes)=}")
        return np.abs(wt_clashes - mut_clashes)

    def mutate_pdb(self):

        cmd = pymol.cmd
        # cmd.feedback("disable", "all", "actions")  # Disable feedback for all actions
        # cmd.reinitialize()
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pdbfile = self.pdb_file_wt_path
            aa_mut = self.mut
            aa_wt = self.wt
            pos = self.pos

            cmd.reinitialize("everything")
            cmd.feedback("disable", "all", "actions")
            correct_modelname = "FLNc"
            # print(f"we load {pdbfile}")
            cmd.load(pdbfile)
            cmd.wizard("mutagenesis")
            cmd.do("refresh_wizard")
            idx = str(pos)
            mutation_trgt = aa_mut.upper()  # HERE INSERT THE CORRECT AMINO ACID
            # print(mutation_trgt)
            cmd.get_wizard().set_mode(mutation_trgt)
            cmd.get_wizard().do_select(idx + "/")

            trgt = cmd.get_wizard().bump_scores
            try:
                trgt_2 = sum(trgt) / len(trgt)
            except:
                # then there is no trgt aka 0.
                trgt_2 = 0

            # print(trgt)
            # print(dir(cmd.get_wizard()))
            # cmd.get_wizard().apply()
            # print(trgt_check)
            rot = 0
            rot_ex = 0
            rot_super_strict = 0
            sum_strain = 0
            for strain in trgt:
                sum_strain += strain
                if strain > 44:
                    rot += 1
                    # print("found at least one suitable rotamere at cutoff 44 strain!")
                if strain > 30:
                    rot_ex += 1
                if strain > 23:  # makes most sense and is strict enough
                    rot_super_strict += 1

            # print("Total amount of rotameres found below cutoff of 44 strain:", rot)
            # print("Total amount of rotameres found below cutoff of 30 strain:", rot_ex)
            # print("Total amount of rotameres found below cutoff of 25 strain:", rot_super_strict)

            # entry_form.r1clash.select()
            try:
                index_hit = trgt.index(min(trgt))  # fish out the best hit
                cmd.frame("0")  # always take 0 !!! this is the most probable hit.
            except:
                index_hit = 0
                cmd.frame("0")

            # print(index_hit)
            # print("here afterwards it still works")
            cmd.get_wizard().apply()
            cmd.set_wizard()

            mut_pdb_save = f"./FLNC_mutated_pdbs/FLNC_{aa_wt}{pos}{aa_mut}.pdb"  # We generate a mutant pdb file for the user #not correct
            # print(correct_modelname)
            cmd.save(mut_pdb_save, correct_modelname)  # saved into the correct directory.
            cmd.reinitialize("everything")
            return rot, rot_ex, rot_super_strict, trgt_2, mut_pdb_save

        finally:
            sys.stdout = original_stdout

    def load_AMIVA_F(self):
        # here we load our pretrained Model for our predictions.
        loaded_model = None

        try:

            loaded_model = load('./Pretrained_AMIVA_F/AMIVA_F.joblib')

        except Exception as e:
            print(e)

        self.AMIVA_F = loaded_model

    def clashcheck(self, mut=None):

        # if called default will compute clashes for the wildtype structure.
        # if called with mut =True will use the mutated structure.

        pdbfile = self.pdb_file_wt_path if mut == None else self.mut_pdb_save
        pos = self.pos

        parser = PDBParser(QUIET=True)
        structureclash = parser.get_structure("FLNc", pdbfile)

        target_residue = int(pos)
        atoms_of_interest = list(structureclash[0]["A"][target_residue])
        atoms_interest_without_mainchain = []
        # print(len(atoms_of_interest))
        for atom in atoms_of_interest:
            if atom.get_name() != "N" and atom.get_name() != "O" and atom.get_name() != "C" and atom.get_name() != "CA":
                atoms_interest_without_mainchain.append(atom)

        target_atoms = [atom.get_coord() for atom in
                        atoms_interest_without_mainchain]  # all atoms of residue that gets modified posttranslationally

        # print(len(target_atoms))
        search_atoms = Selection.unfold_entities(structureclash, "A")  # load in all atoms of structure

        ns = NeighborSearch(search_atoms)

        hits_contacts = []
        hits_clashes = []
        for check_atoms in target_atoms:
            proximal_atoms = ns.search(check_atoms, 5, "A")  # we define contacts as below 4 Angström radius cutoff
            for matches in proximal_atoms:
                hits_contacts.append(matches)

        for check_atoms in target_atoms:
            clash_atoms = ns.search(check_atoms, 3, "A")  # we define clashes as below 2.5 Angström radius cutoff
            for matches in clash_atoms:
                hits_clashes.append(matches)

        # print((len(hits_clashes),len(hits_contacts)))
        clashset = set(hits_clashes)
        contactset = set(hits_contacts)
        clashresidues = []
        contactresidues = []
        clashcounter = 0
        contactcounter = 0
        for a in clashset:
            # print((a.get_parent().get_resname(), a.get_parent().get_id()[1], a.get_name()))
            clashresidues.append(a.get_parent().get_id()[1])
        for a in contactset:
            # print((a.get_parent().get_resname(), a.get_parent().get_id()[1], a.get_name()))
            contactresidues.append(a.get_parent().get_id()[1])
        while target_residue in clashresidues:  # removal of hits that are the target atom itself
            clashresidues.remove(int(target_residue))
        while target_residue in contactresidues:
            contactresidues.remove(int(target_residue))
        # print(len(clashresidues))
        # for hits in clashresidues:
        #    print(hits)
        for hits in contactresidues:
            # print(hits)
            contactcounter += 1
        for hits in clashresidues:
            # print(hits)
            contactcounter += 1
            clashcounter += 1
        # print((len(clashset),len(contactset)))
        # print(str(contactcounter) + " contacts", str(clashcounter) + " clashes")
        return (contactcounter, clashcounter)

    def setup_cost_dic(self):

        mut_df_location = self.mut_df

        mut_df = pd.read_csv(mut_df_location)

        mut_df["position"] = mut_df.iloc[:, 0].str[3:-3].astype(int)
        mut_df["wt_aa"] = mut_df.iloc[:, 0].str[0:3]
        mut_df["mut_aa"] = mut_df.iloc[:, 0].str[-3:]

        analysis_dict = defaultdict(list)
        for aa_mut, aa_wt, effect in zip(mut_df["wt_aa"], mut_df["mut_aa"], mut_df["effect"]):
            analysis_dict[f"{aa_wt}_{aa_mut}"].append(effect)

        sorted_dict = dict(sorted(analysis_dict.items()))

        total_tolerance = []
        for keys, vals in sorted_dict.items():
            freq_count = Counter(vals)
            for item, count in freq_count.items():
                total_tolerance.append((keys, item, count))

        counter_dict = defaultdict(lambda: defaultdict(int))

        # Iterate through the data and update counts
        for entry in total_tolerance:
            amino_acid, status, count = entry[0].split('_')[0], entry[1], entry[2]
            counter_dict[amino_acid][status] += count
            counter_dict[amino_acid]["total"] += count

        # Create a pandas DataFrame from the nested dictionary
        df = pd.DataFrame.from_dict(counter_dict, orient="index")

        # Calculate relative frequencies
        for amino_acid in df.index:
            total_pathological_count = df.loc[amino_acid, 'pathological']
            total_count = df.loc[amino_acid, 'total']
            df.loc[amino_acid, 'frequencies'] = total_pathological_count / total_count
            for status in ['benign', 'pathological']:
                df.loc[amino_acid, f'relative_{status}_frequency'] = df.loc[amino_acid, status] / total_count

        # Create a matrix DataFrame for relative pathological frequencies
        matrix_df = df[['relative_pathological_frequency']].rename(
            columns={'relative_pathological_frequency': 'pathological_propensity'})

        # Display the result
        # print(matrix_df)

        matrix_df.to_csv("./Supplementary/Aa_matrix.csv")

    def visualize_indiviual_contributions(self):

        import shap
        import matplotlib.pyplot as plt

        X = self.input
        predicted_class = self.predicted_class
        probability = self.probability
        self.explainer = shap.TreeExplainer(self.AMIVA_F)
        self.shap_values = self.explainer.shap_values(self.input)

        prediction_overview = self.analysis_input
        explainer = self.explainer
        shap_values = self.shap_values

        shap_values_for_sample = shap_values[1][0, :]

        shap_df = pd.DataFrame({'Feature': X.columns, 'SHAP Value': shap_values_for_sample})

        # Plot horizontal bar chart with different colors for positive and negative values
        plt.figure(figsize=(10, 6))  # Adjust figure size if needed
        colors = shap_df['SHAP Value'].apply(lambda x: 'skyblue' if x <= 0 else 'red')
        bars = plt.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)

        # Custom Legend
        legend_labels = {'Benign contribution': 'skyblue', 'Pathogenic contribution': 'red'}
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for label, color in
                          legend_labels.items()]

        plt.legend(handles=legend_handles, loc='upper right')

        plt.title(
            f'AMIVA_F Feature Importance for Prediction: {prediction_overview}\nPrediction: {predicted_class}\nProbability: {probability}',
            fontsize=16)
        plt.xlabel('SHAP Value')
        plt.ylabel('Feature')

        plt.savefig(f'./Predictions/Shap_plot_{prediction_overview}.png')

        #plt.show()

    def cost_matrix(self):

        df_path = self.mut_df
        analysis_dict = defaultdict(list)

        mut_df = pd.read_csv(df_path, sep=",")

        mut_df["position"] = mut_df.iloc[:, 0].str[3:-3].astype(int)
        mut_df["wt_aa"] = mut_df.iloc[:, 0].str[0:3]
        mut_df["mut_aa"] = mut_df.iloc[:, 0].str[-3:]

        for aa_mut, aa_wt, effect in zip(mut_df["wt_aa"], mut_df["mut_aa"], mut_df["effect"]):
            analysis_dict[f"{aa_wt}_{aa_mut}"].append(effect)

        sorted_dict = dict(sorted(analysis_dict.items()))

        total_tolerance = []
        for keys, vals in sorted_dict.items():
            freq_count = Counter(vals)
            for item, count in freq_count.items():
                total_tolerance.append((keys, item, count))

        # print(total_tolerance)

        # Create a nested defaultdict to store the counts
        def transition_score(conversion_data):
            """
            Calculate a combined score for each potential transition based on benign and pathological transitions.

            Parameters:
            - conversion_data: A list of tuples containing conversion data.

            Returns:
            - A dictionary mapping each potential transition to its combined score.
            """
            scores = {}

            for amino_acid, state, count in conversion_data:
                key = (amino_acid, state)
                if key not in scores:
                    scores[key] = 0

                # Assign scores based on the total number of transitions
                if state == 'benign':
                    scores[
                        key] -= 2.5 * count
                elif state == 'pathological':
                    scores[key] += count

            # Combine scores for each amino acid
            combined_scores = {}
            for key, score in scores.items():
                amino_acid = key[0]
                if amino_acid not in combined_scores:
                    combined_scores[amino_acid] = 0
                combined_scores[amino_acid] += score

            return combined_scores

        # Calculate combined scores for the given data
        combined_scores = transition_score(total_tolerance)

        return combined_scores

    def SAP_score_2(self, radius=10):

        pdbfile = self.pdb_file_wt_path
        pos = self.pos

        with open("./Supplementary/Hydrophobicity_scale_black_mold.txt", "r") as hyd_in:

            normal_dict = defaultdict()
            for lines in hyd_in:
                lines = lines.replace("\n", "")
                lines = lines.split("\t")
                aa, val = lines[0].upper(), float(lines[1])
                normal_dict[aa] = val

            updated_hydrophobicity = defaultdict()

            scaler_val = normal_dict["GLY"]

            for keys, vals in normal_dict.items():
                updated_hydrophobicity[keys] = vals - scaler_val

        rel_sasa_dict = defaultdict()

        # this will contain the rel SASA of the sidechains of respective residues
        with open("./Supplementary/Relative_SASA_Values.txt", "r") as rel_sasa_in:
            for lines in rel_sasa_in:
                full_info = lines.split("\t")
                rel_sasa_dict[full_info[0].upper()] = full_info[4]

        probe_radius_val = 1.4
        resolution_val = 20
        # print(f"thi sis pdb file {pdbfile}")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("default", pdbfile)
        # print("we grabbed the structure successfully")

        # relative sasa taken from : https://www.sciencedirect.com/science/article/pii/S1476927114001698

        freesasa_struct = freesasa.structureFromBioPDB(structure,
                                                       freesasa.Classifier.getStandardClassifier('naccess'))

        result = freesasa.calc(freesasa_struct, freesasa.Parameters(
            {'algorithm': freesasa.ShrakeRupley, 'probe-radius': float(probe_radius_val),
             'n-points': float(resolution_val)}))

        atom_list = Bio.PDB.Selection.unfold_entities(structure, "A")

        atom_coords = [(atom.get_coord(), atom.get_name(), result.atomArea(i), atom.get_parent().get_resname(),
                        atom.get_parent().get_id()[1]) for i, atom in enumerate(atom_list)]

        # atom_coords = (coords, atom_name, SSA, parent_residue)
        # print(f"we have atom coords {atom_coords}")
        # print(atom_coords)
        searching_for_target = []
        full_info_lst = []
        abs_sasa_dict = defaultdict()
        # print(f"filter out our hits.")
        for atom_coords, atom_name, area_abs, restype, resnumber in atom_coords:
            full_info_lst.append((atom_coords, atom_name, area_abs, rel_sasa_dict[restype], restype, resnumber))
            abs_sasa_dict[(resnumber, atom_name)] = area_abs
            if resnumber == int(pos):
                searching_for_target.append(
                    (atom_coords, atom_name, area_abs, rel_sasa_dict[restype], restype, resnumber))
        ns = Bio.PDB.NeighborSearch(atom_list)
        # print(searching_for_target)
        # print(abs_sasa_dict)
        main_chain = ["N", "CA", "C", "O"]
        total_sum_sap = 0
        # print(f"we search now for surr and this is searching_for_target {searching_for_target}")
        for atom_coords, atom_name, area_abs, rel_sasa_parent, restype, resnumber in searching_for_target:
            # f'''For each atom we will make a search for all surrounding atoms that are within {cutoff} A radius.'''
            # returns the residue number.
            proximal_atoms = ns.search(atom_coords, 10, "A")
            sum_terms = 0
            for atoms in proximal_atoms:
                residue_id = atoms.get_parent().get_id()[1]
                residue_name = atoms.get_parent().get_resname()
                search_tuple = (residue_id, atoms.get_name())
                if atoms.get_name() not in main_chain:
                    abs_sasa_atom = abs_sasa_dict[search_tuple]
                    rel_sasa_res = float(rel_sasa_dict[residue_name])
                    hydrophobicity_res = updated_hydrophobicity[residue_name]
                    # print(abs_sasa_atom)
                    # print(rel_sasa_res)
                    # print(hydrophobicity_res)
                    sum_term = ((abs_sasa_atom / rel_sasa_res) * hydrophobicity_res)
                    sum_terms += sum_term
                    # print(abs_sasa_atom, rel_sasa_res,hydrophobicity_res, residue_id)
            total_sum_sap += sum_terms
        # print(f"this is total sum {total_sum_sap} and this is len searching for target: {len(searching_for_target)}")
        # print(total_sum_sap)
        # divide by num atoms per residue to get an overall residue based score.
        return total_sum_sap / len(searching_for_target)

    def get_ddg(self):

        pos = self.pos
        aa_wt = self.wt
        aa_mut = self.mut
        chain = "A"
        reverse_trans_aa_dict = {"ALA": "A", "GLY": "G", "SER": "S", "THR": "T", "CYS": "C", "MET": "M", "VAL": "V",
                                 "ILE": "I",
                                 "LEU": "L", "PHE": "F", "TYR": "Y", "TRP": "W", "HIS": "H", "GLN": "Q", "GLU": "E",
                                 "ASN": "N",
                                 "ASP": "D", "PRO": "P", "LYS": "K", "ARG": "R"}

        domain_name = self.domain_info

        input_line = None

        with open("./Korpm/input.txt", "w") as fh_out:
            fh_out.write(
                f"{domain_name} {reverse_trans_aa_dict[aa_wt.upper()]}{chain}{pos}{reverse_trans_aa_dict[aa_mut.upper()]}")

        # now lets run korpm.
        try:
            cmd_to_run = f"./Korpm/sbg/bin/korpm_gcc ./Korpm/input.txt --dir ./Korpm/testdir --score_file ./Korpm/pot/korp6Dv1.bin -o ./Korpm/testresult_out.txt"
            cmd_to_run = cmd_to_run.split(" ")

            result = subprocess.run(cmd_to_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        except Exception as e:
            print("Korpm did not work")
            print(e)

        with open("./Korpm/testresult_out.txt", "r") as fh_result:
            for lines in fh_result:
                lines = lines.replace("\n", "")
                lines = lines.split(" ")
                result = lines[-1]

        return result

    def aa_change_analysis(self, save=False):

        # is this the proper df ? needs to be checked.
        df = self.mut_df

        analysis_dict = defaultdict(list)

        for aa_mut, aa_wt, effect in zip(df["wt_aa"], df["mut_aa"], df["effect"]):
            analysis_dict[f"{aa_wt}_{aa_mut}"].append(effect)

        sorted_dict = dict(sorted(analysis_dict.items()))

        total_tolerance = []

        for keys, vals in sorted_dict.items():
            # print(keys)
            freq_count = Counter(vals)
            for item, count in freq_count.items():
                # print(item, count)
                total_tolerance.append((keys, item, count))

        # print(total_tolerance)

        counter_dict = defaultdict(lambda: defaultdict(int))

        # Iterate through the data and update counts
        for entry in total_tolerance:
            amino_acid, status, count = entry[0].split('_')[0], entry[1], entry[2]
            counter_dict[amino_acid][status] += count
            counter_dict[amino_acid]["total"] += count

        # Create a pandas DataFrame from the nested dictionary
        df = pd.DataFrame.from_dict(counter_dict, orient="index")

        # Calculate relative frequencies
        for amino_acid in df.index:
            total_pathological_count = df.loc[amino_acid, 'pathological']
            total_count = df.loc[amino_acid, 'total']
            df.loc[amino_acid, 'frequencies'] = total_pathological_count / total_count
            for status in ['benign', 'pathological']:
                df.loc[amino_acid, f'relative_{status}_frequency'] = df.loc[amino_acid, status] / total_count

        # Create a matrix DataFrame for relative pathological frequencies
        matrix_df = df[['relative_pathological_frequency']].rename(
            columns={'relative_pathological_frequency': 'pathological_propensity'})

        if save:
            matrix_df.to_csv("./Supplementary/Aa_matrix.csv")

        return matrix_df

    def conservation_func(self):

        pos = self.pos
        domain = self.domain_num
        domain_offset = self.domain_offset

        pos_dict = defaultdict()

        # continue here!
        with open(f"./Conservation_Files/conservation_scores_ref_{domain}.csv",
                  "r") as con_in:
            for lines in con_in:
                lines = lines.split(",")
                # print(lines)
                if lines[1] == "shannon":
                    continue
                #print(lines)
                position = lines[0]
                score_shannon = lines[1]
                score_relative = lines[2]
                score_lockless = lines[3].replace("\n", "")

                # print(f"position:{position}, pos:{pos}, offset:{domain_offset}")
                pos_dict[str(int(position) + int(domain_offset) - 1)] = (score_shannon, score_relative, score_lockless)

            return pos_dict.get(str(pos), (None, None, 2.76149241879783))


    def size_x_hydro(self):

        aa_wt = self.wt
        aa_mut = self.mut

        reverse_trans_aa_dict = {"ALA": "A", "GLY": "G", "SER": "S", "THR": "T", "CYS": "C", "MET": "M", "VAL": "V",
                                 "ILE": "I",
                                 "LEU": "L", "PHE": "F", "TYR": "Y", "TRP": "W", "HIS": "H", "GLN": "Q", "GLU": "E",
                                 "ASN": "N",
                                 "ASP": "D", "PRO": "P", "LYS": "K", "ARG": "R"}

        if len(aa_wt) or len(aa_mut) != 1:
            try:
                aa_wt = reverse_trans_aa_dict[aa_wt.upper()]

                aa_mut = reverse_trans_aa_dict[aa_mut.upper()]
            except Exception as e:
                raise ValueError(
                    "Inappropiate input format! Wildtype and Mutated amino acid need \nto either be provided "
                    "as 3 letter or 1 letter code e.g ALA or A") from e

        hydrophobchange = {"G": -0.4, "A": 1.8, "V": 4.2, "I": 4.5, "L": 3.8, "M": 1.9, "F": 2.8,
                           "Y": -1.3,
                           "W": -0.9, "P": -1.6, "C": 2.5, "Q": -3.5, "N": -3.5, "T": -0.7, "S": -0.8,
                           "E": -3.5, "D": -3.5, "K": -3.9, "H": -3.2, "R": -4.5}

        num_atoms_lib = {"G": 10, "A": 13, "V": 19, "I": 22, "L": 22, "M": 20, "F": 23,
                         "Y": 24,
                         "W": 27, "P": 17, "C": 14, "Q": 20, "N": 17, "T": 17, "S": 14,
                         "E": 19, "D": 16, "K": 24, "H": 20, "R": 26}

        weight_lib = {"G": 75, "A": 89, "V": 117, "I": 131, "L": 131, "M": 149, "F": 165,
                      "Y": 181,
                      "W": 204, "P": 115, "C": 121, "Q": 146, "N": 132, "T": 119, "S": 105,
                      "E": 147, "D": 133, "K": 146, "H": 155, "R": 174}

        dhydrophob = round(hydrophobchange[aa_mut] - hydrophobchange[aa_wt],
                           2)  # pos if more hydrophobic neg if more hydrophilic
        size_change = num_atoms_lib[aa_mut] - num_atoms_lib[aa_wt]  # pos if larger aa inserted neg if smaller
        weight_change = weight_lib[aa_mut] - weight_lib[aa_wt]  # pos if heavier aa inserted neg if smaller

        return dhydrophob, weight_change, size_change  # dhydrophob **3 * weight_change - > hydro_size_change_abs 	corr 0.507459

    def get_propensity(self):

        aa_wt = self.wt
        aa_mut = self.mut

        df = pd.read_csv("./Supplementary/Aa_matrix.csv")

        # df.drop(columns="Unnamed: 0", inplace=True)
        # print(df.head(20))
        row_loc = {'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLY': 6,
                   'HIS': 7, 'ILE': 8, 'LEU': 9, 'LYS': 10, 'MET': 11, 'PHE': 12, 'PRO': 13, 'SER': 14, 'THR': 15,
                   'TRP': 16,
                   'VAL': 17, 'GLU': 18, 'TYR': 19, 'ALA': 0}

        return df.loc[row_loc[aa_wt.upper()], "pathological_propensity"]

    def _infer_wildtype(self):

        # could be inplemented if user submits wrong aa.
        pos = self.pos
        pass

    def retrieve_PDB_and_domain(self):

        pos = self.pos

        if int(pos) > 35 and int(pos) < 263:
            pdbfile1 = "./FLNC_Structures/FLNc_ABD.pdb"
            domaininfo = "FLNc_ABD"
            domain_num = 0
            domain_offset = 36
        elif int(pos) >= 270 and int(pos) <= 368:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_1.pdb"
            domaininfo = "FLNc_domain_1"
            domain_num = 1
            domain_offset = 270
        elif int(pos) >= 370 and int(pos) <= 468:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_2.pdb"
            domaininfo = "FLNc_domain_2"
            domain_num = 2
            domain_offset = 370
        elif int(pos) >= 469 and int(pos) <= 565:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_3.pdb"
            domaininfo = "FLNc_domain_3"
            domain_num = 3
            domain_offset = 469

        elif int(pos) >= 568 and int(pos) <= 760:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_4_and_5.pdb"
            domaininfo = "FLNc_domain_4_and_5"
            if int(pos) <= 658:
                domain_num = 4
                domain_offset = 568
            else:
                domain_num = 5
                domain_offset = 658
        elif int(pos) >= 761 and int(pos) <= 861:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_6.pdb"
            domaininfo = "FLNc_domain_6"
            domain_num = 6
            domain_offset = 761
        elif int(pos) >= 862 and int(pos) <= 960:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_7.pdb"
            domaininfo = "FLNc_domain_7"
            domain_num = 7
            domain_offset = 862
        elif int(pos) >= 961 and int(pos) <= 1056:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_8.pdb"
            domaininfo = "FLNc_domain_8"
            domain_num = 8
            domain_offset = 961
        elif int(pos) >= 1057 and int(pos) <= 1149:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_9.pdb"
            domaininfo = "FLNc_domain_9"
            domain_num = 9
            domain_offset = 1057
        elif int(pos) >= 1150 and int(pos) <= 1244:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_10.pdb"
            domaininfo = "FLNc_domain_10"
            domain_num = 10
            domain_offset = 1150
        elif int(pos) >= 1245 and int(pos) <= 1344:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_11.pdb"
            domaininfo = "FLNc_domain_11"
            domain_num = 11
            domain_offset = 1245
        elif int(pos) >= 1345 and int(pos) <= 1437:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_12.pdb"
            domaininfo = "FLNc_domain_12"
            domain_num = 12
            domain_offset = 1345
        elif int(pos) >= 1438 and int(pos) <= 1533:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_13.pdb"
            domaininfo = "FLNc_domain_13"
            domain_num = 13
            domain_offset = 1438
        elif int(pos) >= 1535 and int(pos) <= 1736:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_14_and_15.pdb"
            domaininfo = "FLNc_domain_14_and_15"
            if int(pos) <= 1630:
                domain_num = 14
                domain_offset = 1535
            else:
                domain_num = 15
                domain_offset = 1631
        elif int(pos) >= 1759 and int(pos) <= 1853:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_16.pdb"
            domaininfo = "FLNc_domain_16"
            domain_num = 16
            domain_offset = 1759
        elif int(pos) >= 1854 and int(pos) <= 1946:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_17.pdb"
            domaininfo = "FLNc_domain_17"
            domain_num = 17
            domain_offset = 1854
        elif int(pos) >= 1947 and int(pos) <= 2033:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_18.pdb"
            domaininfo = "FLNc_domain_18"
            domain_num = 18
            domain_offset = 1947
        elif int(pos) >= 2036 and int(pos) <= 2128:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_19.pdb"
            domaininfo = "FLNc_domain_19"
            domain_num = 19
            domain_offset = 2036
        elif int(pos) >= 2129 and int(pos) <= 2315:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_20.pdb"
            domaininfo = "FLNc_domain_20"
            domain_num = 20
            domain_offset = 2129
        elif int(pos) >= 2316 and int(pos) <= 2401:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_21.pdb"
            domaininfo = "FLNc_domain_21"
            domain_num = 21
            domain_offset = 2316
        elif int(pos) >= 2402 and int(pos) <= 2509:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_22.pdb"
            domaininfo = "FLNc_domain_22"
            domain_num = 22
            domain_offset = 2402
        elif int(pos) >= 2510 and int(pos) <= 2598:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_23.pdb"
            domaininfo = "FLNc_domain_23"
            domain_num = 23
            domain_offset = 2510
        elif int(pos) >= 2633 and int(pos) <= 2725:
            pdbfile1 = "./FLNC_Structures/FLNc_domain_24.pdb"
            domaininfo = "FLNc_domain_24"
            domain_num = 24
            domain_offset = 2633
        else:
            return None, None, None, None

        return pdbfile1, domaininfo, domain_num, domain_offset

    def new_feature_size_x_hydro(self):

        aa_wt = self.wt
        aa_mut = self.mut

        reverse_trans_aa_dict = {"ALA": "A", "GLY": "G", "SER": "S", "THR": "T", "CYS": "C", "MET": "M", "VAL": "V",
                                 "ILE": "I",
                                 "LEU": "L", "PHE": "F", "TYR": "Y", "TRP": "W", "HIS": "H", "GLN": "Q", "GLU": "E",
                                 "ASN": "N",
                                 "ASP": "D", "PRO": "P", "LYS": "K", "ARG": "R"}

        hydrophobchange = {"G": -0.4, "A": 1.8, "V": 4.2, "I": 4.5, "L": 3.8, "M": 1.9, "F": 2.8,
                           "Y": -1.3,
                           "W": -0.9, "P": -1.6, "C": 2.5, "Q": -3.5, "N": -3.5, "T": -0.7, "S": -0.8,
                           "E": -3.5, "D": -3.5, "K": -3.9, "H": -3.2, "R": -4.5}

        num_atoms_lib = {"G": 10, "A": 13, "V": 19, "I": 22, "L": 22, "M": 20, "F": 23,
                         "Y": 24,
                         "W": 27, "P": 17, "C": 14, "Q": 20, "N": 17, "T": 17, "S": 14,
                         "E": 19, "D": 16, "K": 24, "H": 20, "R": 26}

        weight_lib = {"G": 75, "A": 89, "V": 117, "I": 131, "L": 131, "M": 149, "F": 165,
                      "Y": 181,
                      "W": 204, "P": 115, "C": 121, "Q": 146, "N": 132, "T": 119, "S": 105,
                      "E": 147, "D": 133, "K": 146, "H": 155, "R": 174}

        if len(aa_wt) == 3 and len(aa_mut) == 3:
            aa_wt = reverse_trans_aa_dict[aa_wt.upper()]
            aa_mut = reverse_trans_aa_dict[aa_mut.upper()]

        dhydrophob = round(hydrophobchange[aa_mut] - hydrophobchange[aa_wt],
                           2)  # pos if more hydrophobic neg if more hydrophilic
        size_change = num_atoms_lib[aa_mut] - num_atoms_lib[aa_wt]  # pos if larger aa inserted neg if smaller
        weight_change = weight_lib[aa_mut] - weight_lib[aa_wt]  # pos if heavier aa inserted neg if smaller

        return dhydrophob, weight_change, size_change  # dhydrophob **3 * weight_change - > hydro_size_change_abs 	corr 0.507459

    def prediction_input_prep(self):

        ddg = self.ddg
        ddg_abs = np.abs(float(ddg))

        sap = self.SAP
        cost = self.get_cost()
        lockless_cons = self.conservation[2]
        hydro_x_size_x_weight = self.new_feature_size_x_hydro()
        hydro_x_size_x_weight = hydro_x_size_x_weight[0] * hydro_x_size_x_weight[1] * hydro_x_size_x_weight[2]

        # this is abs clashdiff!
        clashdiff = self.clashcounter_diff

        # [["cost", 'hydro_weight_num_abs_dhydro', "lockless_cons", "ddg_abs", "clashcounter_diff", "SAP_scores"]]

        col_names = ["cost", "hydro_weight_num_abs_dhydro",
                     "lockless_cons", "ddg_abs", "clashes_introduced",
                     "SAP_scores"]

        col_vals = [cost, hydro_x_size_x_weight, lockless_cons, ddg_abs,
                    clashdiff, sap]

        df_input = pd.DataFrame([col_vals], columns=col_names)

        # this is our input.
        return df_input

    def preprocessing(self):
        # Call the method and store the results
        self.pdb_file_wt_path, self.domain_info, \
            self.domain_num, self.domain_offset = self.retrieve_PDB_and_domain()

        self.dhydrophob, self.weight_change, self.size_change = self.size_x_hydro()

        self.SAP = self.SAP_score_2()

        self.conservation = self.conservation_func()

        self.propensity = self.get_propensity()

        self.combined_scores = self.cost_matrix()

        self.rot, self.rot_ex, self.rot_super_stric, self.trgt_2, self.mut_pdb_save = self.mutate_pdb()

        self.ddg = self.get_ddg()

        self.cost = self.get_cost()

        self.clash_wt = self.clashcheck(mut=None)
        self.clash_mut = self.clashcheck(mut=True)

        self.dhydro_dsize_dweigth = self.new_feature_size_x_hydro()

        self.clashcounter_diff = self._compute_clashcountdiff()

        self.analysis_input = f"{self.wt} {self.pos} {self.mut}"

    def get_cost(self):
        # helper function to retrieve the cost from the cost_dictionary
        aa_wt = self.wt
        aa_mut = self.mut
        cost_dic = self.combined_scores

        target = aa_wt + "_" + aa_mut
        try:
            cost = cost_dic[target]
        except:
            # we have no info then
            cost = 0

        return cost

    def calculate_SASA(self, structure: str, resid: int):

        structure = self.pdb_file_wt_path
        resid = self.pos

        parser = PDBParser(QUIET=True)

        structure = parser.get_structure("placeholder_name", f"{structure}")

        result, residue_areas = freesasa.calcBioPDB(structure)

        res2 = freesasa.Result.residueAreas(result)
        for chains, dics in res2.items():
            for keys, vals in dics.items():
                if keys == str(resid):
                    return vals.total, vals.relativeTotal
                    # print(keys, vals.total, vals.relativeTotal)


def main():
    try:
        parser = argparse.ArgumentParser(
            description='AMIVA_F: Predict the effect of single point missense mutations \n'
                        'on pathogenicity in Filamin C',
            epilog="Example usage: python AMIVA_F.py MET 82 LYS --visualization True")
        parser.add_argument('wildtype', type=str, help='Wildtype amino acid')
        parser.add_argument('position', type=int, help='Amino acid position')
        parser.add_argument('mutation', type=str, help='Mutation amino acid')
        parser.add_argument('--visualization', type=str, help='Type of visualization')

        args = parser.parse_args()

        wildtype = args.wildtype
        position = args.position
        mutation = args.mutation
        visualize = args.visualization

        if visualize == "yes" or visualize == "True" or visualize == "TRUE":
            visualize = True

        print(f'Wildtype: {wildtype}')
        print(f'Position: {position}')
        print(f'Mutation: {mutation}')
        print(f'Visualization Type: {visualize}')

    except Exception as e:
        print(e)

    # initialize the PDB object
    PDB_object = PDB_setup(wildtype, mutation, position, visualize=visualize)
    # set path domain info and other variables according to user submitted amino acid position.

    PDB_object.preprocessing()
    # grab amiva_F to make predictions.
    PDB_object.load_AMIVA_F()
    # now make the prediction
    PDB_object.prediction_input_prep()
    # now lets see
    PDB_object.probability_pred()

    # individual info about the features contributing to the individual prediction.
    if visualize == True:
        PDB_object.visualize_indiviual_contributions()


if __name__ == "__main__":
    main()
