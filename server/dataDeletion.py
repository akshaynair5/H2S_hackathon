import os
from dotenv import load_dotenv
from firebase_admin import firestore
from database import db  
from datetime import datetime


load_dotenv()

NORM_IDS_TO_DELETE = [
    "8e606cfafcb931852464a866e2e08c5bde352eed971db87bdd9b0c5a240cc6fd",
    "2a8dc05a317f851dda55054db5db37904a470be2306ff56ef78946e774624b3a",
    "acc35971e09306bb8469673c1dd2bee13671f770ee6fdcdf4c5bea3bf219ddd6",
    "bfe196c83aea2c89feab0a62d4639bd33276791f6e68676403e2cbe051cbb343",
    "4231f18aa959083e072fb9149e96a98beba873b734c82b23974eb2fe588a69b8",
    "16a2540305bbbb81cdb3d372911fcf726a89a1d654947b377da990169d8b95f2",
    "ede3c3609c0f24eff4a0c23519688a67fc58e16ea3347e1f8d86f19c7e3115a4",
    "c9138009d8a83421f3bb132819250e1e4f6eeddb6217abedef2bfc87a3e255af",
    "2676c1b78275ea8110ede1f33af693b1e4549aa4f10770dcf2fcd4d6f4850aee",
    "9291feda999d6e9534c9cfff5ede4d730cda7f85c05b64e1d8086823e016cff3",
]


COLLECTION_NAME = "articles"  
DRY_RUN = True  


def bulk_delete_by_norm_ids(norm_ids, collection_name=COLLECTION_NAME, dry_run=DRY_RUN):
    if not norm_ids:
        print("No norm_ids provided. Exiting.")
        return

    print(f"Starting bulk deletion for {len(norm_ids)} norm_id(s) in '{collection_name}' collection...")
    print(f"Dry run: {'YES' if dry_run else 'NO'}")

    total_found = 0
    total_deleted = 0
    deleted_doc_ids = []

    for norm_id in norm_ids:
        print(f"\nSearching for normalized_id == '{norm_id}'")
        try:
            
            docs = db.collection(collection_name) \
                     .where("normalized_id", "==", norm_id) \
                     .stream()

            doc_list = list(docs)
            if not doc_list:
                print(f"  No documents found.")
                continue

            print(f"  Found {len(doc_list)} document(s):")
            for doc in doc_list:
                doc_id = doc.id
                data = doc.to_dict()
                text_snippet = (data.get("text") or "")[:60].replace("\n", " ")
                print(f"    • {doc_id} | Text: '{text_snippet}'...")

                if not dry_run:
                    
                    doc.reference.delete()
                    print(f"      Deleted: {doc_id}")
                else:
                    print(f"      [DRY RUN] Would delete: {doc_id}")

                total_found += 1
                if not dry_run:
                    total_deleted += 1
                    deleted_doc_ids.append(doc_id)

        except Exception as e:
            print(f"  Error querying/deleting for '{norm_id}': {e}")

    
    print("\n" + "="*60)
    if dry_run:
        print(f"DRY RUN COMPLETE")
        print(f"  Would delete: {total_found} document(s)")
    else:
        print(f"BULK DELETION COMPLETE")
        print(f"  Deleted: {total_deleted} document(s)")
    print(f"  Timestamp: {datetime.utcnow().isoformat()}Z")
    print("="*60)

    return {
        "norm_ids_processed": len(norm_ids),
        "total_found": total_found,
        "total_deleted": total_deleted if not dry_run else 0,
        "deleted_doc_ids": deleted_doc_ids
    }



if __name__ == "__main__":
    if not NORM_IDS_TO_DELETE or all(not x.strip() for x in NORM_IDS_TO_DELETE):
        print("Please fill NORM_IDS_TO_DELETE with valid IDs.")
    else:
        results = bulk_delete_by_norm_ids(NORM_IDS_TO_DELETE, dry_run=False)
        print(f"\nFinal Results: {results}")